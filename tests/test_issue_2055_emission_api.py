"""P1c-1 transaction-owned emission vectors and terminal state guards."""
import pytest

from hermes_lcm import ingest_ledger as ledger
from tests.test_ingest_ledger_api import DIGEST, SALT, store as _store


store = _store


def insert(store):
    conn = store._conn
    seq = ledger.next_view_seq(conn)
    ids = ledger.insert_emissions(conn, "lineage", seq, [dict(output_index=1, descriptor_digest=DIGEST, kind="carrier")])
    entries = [ledger.ViewEntry(DIGEST, None, None, None), ledger.ViewEntry(DIGEST, None, None, ids[0], True)]
    ledger.write_view(conn, SALT, "lineage", "emission", "proposed", entries, view_seq=seq)
    return seq, ids, entries


@pytest.mark.parametrize("state", ["adopted", "refused"])
def test_shared_sequence_descriptor_refs_and_terminal_moves(store, state):
    conn = store._conn
    with store.atomic():
        seq, ids, entries = insert(store)
    emission = conn.execute("SELECT emission_id,view_seq,state,next_occ_id,source_store_ids FROM ledger_emissions").fetchone()
    view = ledger.newest_view(conn, "lineage", "emission")
    assert emission == (ids[0], seq, "proposed", None, "[]")
    assert view["view_seq"] == seq and ledger.decode_entries(view["entries"]) == entries
    assert view["entries"][17] == 8
    with pytest.raises(ledger.LedgerFormatError):
        ledger.encode_entries([entries[0]._replace(is_descriptor=True)])
    with store.atomic():
        assert ledger.set_emission_state(conn, "lineage", seq, state) == (1, 1)
    before = (conn.execute("SELECT * FROM ledger_emissions").fetchall(), conn.execute("SELECT * FROM ingest_views").fetchall())
    for repeated in (state, "refused" if state == "adopted" else "adopted"):
        with store.atomic():
            assert ledger.set_emission_state(conn, "lineage", seq, repeated) == (0, 0)
        assert before == (conn.execute("SELECT * FROM ledger_emissions").fetchall(), conn.execute("SELECT * FROM ingest_views").fetchall())


def test_state_move_survives_pruning_and_cannot_touch_other_vector(store):
    conn = store._conn
    with store.atomic():
        first, _, _ = insert(store)
        second, _, _ = insert(store)
        third, _, _ = insert(store)
    assert conn.execute("SELECT view_seq FROM ingest_views ORDER BY view_seq").fetchall() == [(second,), (third,)]
    with store.atomic():
        assert ledger.set_emission_state(conn, "lineage", first, "refused") == (1, 0)
        assert ledger.set_emission_state(conn, "other", third, "adopted") == (0, 0)
    assert conn.execute("SELECT state FROM ledger_emissions ORDER BY view_seq").fetchall() == [("refused",), ("proposed",), ("proposed",)]


def test_emission_helpers_require_owner_and_validate_state(store):
    conn = store._conn
    with pytest.raises(ledger.LedgerTransactionError):
        ledger.insert_emissions(conn, "lineage", 1, [])
    with pytest.raises(ledger.LedgerTransactionError):
        ledger.set_emission_state(conn, "lineage", 1, "adopted")
    with store.atomic(), pytest.raises(ledger.LedgerError):
        ledger.set_emission_state(conn, "lineage", 1, "proposed")
