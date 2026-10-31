"""Real compaction and stub-first pipeline coverage for shadow emission exits."""
import logging

from hermes_lcm import ingest_ledger as ledger
from tests.test_issue_1116_stub_first_exit_commit_proof import (
    committed_session, grow_over_threshold, host_for as _host_for, stub_first_result,
)


host_for = _host_for


def test_real_summary_then_stub_first_emits_the_returned_vector(host_for, caplog):
    caplog.set_level(logging.INFO)
    host = host_for("summary")
    committed_session(host, "summary")
    conn = host.engine._store._conn
    normal = conn.execute("SELECT emission_id,state FROM ledger_emissions ORDER BY emission_id").fetchall()
    assert normal and all(state == "adopted" for _, state in normal)
    grow_over_threshold(host)
    returned, _ = stub_first_result(host, caplog)
    view = ledger.newest_view(conn, host.engine._emission_handle.lineage, "emission")
    entries = ledger.decode_entries(view["entries"])
    fresh = conn.execute("SELECT output_index,emission_id,descriptor_digest FROM ledger_emissions WHERE view_seq=?",
                         (view["view_seq"],)).fetchall()
    assert fresh and view["state"] == "proposed" and len(entries) == len(returned)
    salt = host.engine._store.ingest_ledger_salt()
    for index, emission_id, digest in fresh:
        assert entries[index].is_descriptor and entries[index].ref == emission_id
        assert digest == ledger.identity_digest(salt, returned[index])
