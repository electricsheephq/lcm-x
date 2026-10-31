"""Verified polls must catch metadata damage even if it is later restored."""
import sqlite3

import pytest

from bench.instruments.local_catchup import audit, common, rig
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit",
                  "bench.instruments.local_catchup.tests.test_fix_round2"]
SQL_CONNECT = sqlite3.connect


@pytest.mark.parametrize("table,field", [("lcm_chunk_meta", "char_end"),
                                       ("lcm_embedding_meta", "source_token_count")])
@pytest.mark.parametrize("corrupt", [True, False])
def test_rig_transient_metadata_and_new_rows(corpus, fake_rig, monkeypatch, table, field, corrupt):
    db, manifest, mod = corpus
    a, _, _, _, _, _ = fake_rig
    a.processes, a.rows, a.max_wall, a.restart_fraction = 1, 8, 700, 2
    monkeypatch.setattr(sqlite3, "connect", SQL_CONNECT)
    monkeypatch.setattr(rig, "connect", audit.connect)
    def build(repo, host, target, *args):
        with SQL_CONNECT(db) as src, SQL_CONNECT(target) as dst:
            src.backup(dst)
        return manifest
    monkeypatch.setattr(rig, "build", build)
    calls, finals = [], []
    def poll(target, m, verify=False):
        # Catch-up creates rows after the baseline; the first verified poll records them.
        if not calls:
            fill(target, manifest, mod)
        calls.append(verify)
        damage = corrupt and verify and sum(calls) == 2
        if damage:
            with SQL_CONNECT(target) as c:
                c.execute(f"UPDATE {table} SET {field}={field}+1")
        result = dict(audit.progress(target, m, verify), complete=False)
        if damage:
            with SQL_CONNECT(target) as c:
                c.execute(f"UPDATE {table} SET {field}={field}-1")
        return result
    def final(target, m):
        result = audit.audit(target, m)
        finals.append(result)
        return result
    monkeypatch.setattr(rig, "read_progress", poll)
    monkeypatch.setattr(rig, "audit", final)
    rig.main()
    result = common.load(a.out/"summary.json")
    assert all(r["complete"] and not r["changed_metadata"] for r in finals)
    if corrupt:
        assert len(calls) == 2*result["verify_every_polls"]+1
        assert result["reason"] == "row_or_vector_corruption"
        assert result["state"] == "ABORTED"
    else:
        assert result["reason"] == "max_wall"
        assert not result["per_process_progress"][0]["changed_metadata"]


@pytest.mark.parametrize("preexisting", [True, False])
@pytest.mark.parametrize("table", ["lcm_chunk_meta", "lcm_embedding_meta"])
def test_progress_and_final_audit_preserve_metadata_digest(corpus, preexisting, table):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    if preexisting:
        with audit.connect(db) as c:
            manifest["preimage"] = audit.preimage(c)
    assert not audit.progress(db, manifest, verify=True)["changed_metadata"]
    assert not (manifest["observed_metadata"].keys() & manifest["preimage"]["metadata"].keys())
    with SQL_CONNECT(db) as c:
        c.execute(f"UPDATE {table} SET archived=1")
    assert audit.progress(db, manifest, verify=True)["changed_metadata"]
    final = audit.audit(db, manifest)
    assert final["changed_metadata"] and not final["complete"]
