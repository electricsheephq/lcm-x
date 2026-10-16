"""#1073 review round 2: carry telemetry per assembly, and a linear lineage walk."""
import json

import tests.test_issue_659_user_carry_packet as carry_packet
from tests.test_issue_659_user_carry_packet import _compact, _generated, _history

make = carry_packet.make


def _ending():
    return [{"role": "user", "content": "latest request"}, {"role": "assistant", "content": "ok"}]


def test_an_assembly_without_a_packet_reports_no_delivery(make):
    engine = make()
    out = _compact(engine, [*_history(8), *_ending()])
    assert _generated(engine, out)[1] and engine._last_user_carry["delivered_tokens"] > 0
    engine._assemble_context(None, out[1:], stub_over_cap_tool_results=True)  # forced recovery: never a packet
    assert engine._last_user_carry == {"delivered_tokens": 0}


def _chain(engine, n):
    ids = [f"R{i}" for i in range(n)]
    for newer, older in zip(ids[1:], ids):
        engine._store.write_metadata_json([engine._rotation_predecessor_metadata_key(newer)], json.dumps(older))
    return ids


def _counting_reads(engine, monkeypatch):
    reads, real = [], engine._store.read_metadata_json
    monkeypatch.setattr(engine._store, "read_metadata_json", lambda key: reads.append(key) or real(key))
    return reads


def test_the_lineage_walk_reads_each_hop_once_and_extends_the_previous_chain_by_one_hop(make, monkeypatch):
    engine = make()
    ids = _chain(engine, 3000)
    reads = _counting_reads(engine, monkeypatch)
    engine._session_id = ids[-2]
    assert engine._user_carry_lineage() == ids[-2::-1] and len(reads) <= len(ids)
    reads.clear()
    engine._session_id = ids[-1]  # the next rotation: one read, not a second full walk
    assert engine._user_carry_lineage() == ids[::-1] and len(reads) == 1


def test_a_lineage_cycle_stops_the_walk_with_and_without_a_previous_chain(make):
    engine = make()
    _chain(engine, 4)
    engine._store.write_metadata_json([engine._rotation_predecessor_metadata_key("R0")], json.dumps("R3"))
    engine._session_id = "R2"
    assert engine._user_carry_lineage() == ["R2", "R1", "R0", "R3"]
    engine._session_id = "R3"  # its predecessor R2 holds a chain that already contains R3
    assert engine._user_carry_lineage() == ["R3", "R2", "R1", "R0"]
