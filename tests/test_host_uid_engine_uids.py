"""v0.26.0 slice B2: deterministic engine uids on LCM-generated rows (REVISION 4), the carriers (sites 2, 18
and 19) and the GENERATED shadow class (R3-3 class 2)."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from hermes_lcm import host_uid_emit as emit
from hermes_lcm.engine import _OVERFLOW_RECOVERY_OVERCAP_NOTE
from hermes_lcm.host_uid import host_uid_doctor_lines
from hermes_lcm.reconcile import (
    _COMPACTION_COMMIT_PROOF_METADATA_PREFIX,
    _descriptor_shape_is_well_formed,
    _emission_identity,
    _project_emitted_occurrences,
)
from hermes_lcm.tokens import count_message_tokens
from tests.test_host_uid_shadow import PAD, _counts, _engine, _m, _state_db

engine_uid = getattr(emit, "engine_uid", None)  # absent at the B1 base: red-at-base runs the tests
PERSISTENCE_ONLY = ("message_uid", "_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid",
                    "_row_id", "_db_row_snapshot", "_canonical_row", "timestamp")


@pytest.fixture(autouse=True)
def capable(monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(emit, "_host_uid_capability", True)


def _host(system=False, turns=8):
    rows = [_m("system", "sys", 9.0, "sys")] if system else []
    for i in range(turns):
        rows.append(_m("user", f"question {i}" + PAD, 10.0 + 2 * i, f"u{i}"))
        rows.append(_m("assistant", f"answer {i}" + PAD, 11.0 + 2 * i, f"a{i}"))
    return rows


def _open(tmp_path, session="S", tail=4):
    engine = _engine(tmp_path, session=session)
    engine._hermes_home = str(tmp_path)
    engine._config.fresh_tail_count = tail
    return engine


def _compress(engine, host):
    return engine.compress(copy.deepcopy(host), current_tokens=100_000, force=True)


def _engine_rows(engine):
    return engine._store._conn.execute(
        "SELECT store_id, uid, kind, proof_kind FROM host_uid_bindings WHERE kind = 'engine' ORDER BY rowid").fetchall()


def _provider(rows):
    return [{k: v for k, v in row.items() if k not in PERSISTENCE_ONLY} for row in rows]


def _lineage(engine):
    return engine._host_uid_lineage_key()[0]


# -- mint ---------------------------------------------------------------------------------------------------

def test_engine_uid_formula_is_deterministic_and_distinct():
    uid = engine_uid("tag:root", "summary", "b" * 64, 0)
    expected = "lcmx-engine-uid\0tag:root\0summary\0" + "b" * 64 + "\0" + "0"
    assert uid == hashlib.sha256(expected.encode()).hexdigest()[:32]
    assert len(uid) == 32 and int(uid, 16) >= 0
    variants = {engine_uid("tag:root", "objective", "b" * 64, 0), engine_uid("tag:root", "summary", "c" * 64, 0),
                engine_uid("tag:root", "summary", "b" * 64, 1), engine_uid("tag:other", "summary", "b" * 64, 0)}
    assert uid not in variants and len(variants) == 4


def test_carrier_uid_is_the_summary_engine_uid_and_stable_across_assemblies(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        out = _compress(engine, _host())
        carrier = out[0]
        summary = carrier["content"][:engine._verified_lcm_summary_prefix_end(carrier["content"])]
        assert carrier["message_uid"] == engine_uid(
            _lineage(engine), "summary", hashlib.sha256(summary.encode()).hexdigest(), 0)
        tail = [dict(row) for row in _host()[-4:]]
        again = [engine._assemble_context(None, copy.deepcopy(tail))[0]["message_uid"] for _ in range(2)]
        assert again == [carrier["message_uid"]] * 2
    finally:
        engine.shutdown()


def test_772_cache_and_unchanged_list_keep_one_uid_and_collapse(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        first = _compress(engine, _host())
        persisted = copy.deepcopy(first)
        cache_hits = []
        original = engine._copy_active_replay_messages_preserving_generated_ids
        monkeypatch.setattr(engine, "_copy_active_replay_messages_preserving_generated_ids",
                            lambda *a, **k: cache_hits.append(1) or original(*a, **k))
        count = engine.compression_count
        second = engine.compress(persisted, current_tokens=100)
        assert (engine.compression_count, engine.last_compression_status) == (count, "noop")
        assert cache_hits  # the #772 cached prefix served this call
        assert second is persisted  # L2 §1e: unchanged, with the engine uid taking part in the comparison
        assert second[0]["message_uid"] == first[0]["message_uid"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["not_capable", "off", "no_lineage"])
def test_nothing_minted_without_the_gate_or_a_lineage(tmp_path, monkeypatch, case):
    if case != "no_lineage":
        _state_db(tmp_path, [("S", None, None)])
    if case == "not_capable":
        monkeypatch.setattr(emit, "_host_uid_capability", False)
    if case == "off":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    engine = _open(tmp_path)
    try:
        out = _compress(engine, _host())
        assert "message_uid" not in out[0]
        assert _engine_rows(engine) == [] if engine._store._host_uid_table_exists() else True
        expected = {"role", "content"} | ({"_absorbed_message_uids"} if case == "no_lineage" else set())
        assert set(out[0]) == expected  # the carry still absorbs on a capable host; only the mint needs a root
    finally:
        engine.shutdown()


def test_minting_changes_no_identity_digest_or_token_count(tmp_path, monkeypatch):
    proofs, outputs = [], []
    for capable_host in (True, False):
        monkeypatch.setattr(emit, "_host_uid_capability", capable_host)
        home = tmp_path / str(capable_host)
        home.mkdir()
        _state_db(home, [("S", None, None)])
        engine = _open(home)
        try:
            out = _compress(engine, _host())
            outputs.append(out)
            proofs.append({key: engine._compress_commit_proof[key] for key in
                           ("input", "output", "output_effective", "output_sha256_v3", "effective_sha256_v3")})
            proofs[-1]["emissions"] = [{k: v for k, v in d.items() if k not in {"engine_uid", "scope"}}
                                       for d in engine._compress_commit_proof["emissions"]]
            proofs[-1]["rows"] = [(_emission_identity(m), engine._message_replay_identity(m, strip_carrier=False),
                                   engine._message_replay_identity(m), count_message_tokens(m)) for m in out]
        finally:
            engine.shutdown()
    assert "message_uid" in outputs[0][0] and "message_uid" not in outputs[1][0]
    assert proofs[0] == proofs[1]


def test_provider_bytes_equal_the_off_output(tmp_path, monkeypatch):
    outs = []
    for mode in ("shadow", "off"):
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
        home = tmp_path / mode
        home.mkdir()
        _state_db(home, [("S", None, None)])
        engine = _open(home)
        try:
            outs.append(_compress(engine, _host()))
        finally:
            engine.shutdown()
    assert outs[0] != outs[1] and set(outs[1][0]) == {"role", "content"}
    assert _provider(outs[0]) == _provider(outs[1])


def test_other_generated_rows_get_their_kind(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        lineage = _lineage(engine)
        older = {"role": "user", "content": "OLDER_ASK: summarize the report."}
        newest = {"role": "user", "content": "NEWEST_ASK: rebuild the index " * 300}
        tail = [older, {"role": "assistant", "content": "chatter " * 1200}, newest,
                {"role": "tool", "tool_call_id": "orphan", "content": "status"}]
        note = engine._assemble_overflow_recovery_context(None, tail, assembly_cap_override=120)[-1]
        assert note["content"].startswith(_OVERFLOW_RECOVERY_OVERCAP_NOTE[:20])
        basis = hashlib.sha256(note["content"].encode()).hexdigest()
        assert note["message_uid"] == engine_uid(lineage, "overflow_note", basis, 0)

        marker = engine._fallback_tail_compaction([_m("user", f"r{i}") for i in range(12)])
        omitted = next(row for row in marker if row["content"].startswith("[Context omitted"))
        assert omitted["message_uid"] == engine_uid(
            lineage, "omitted_marker", hashlib.sha256(omitted["content"].encode()).hexdigest(), 0)

        recall = {"role": "user", "content": "recalled memory"}
        monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a, **k: dict(recall))
        rows = engine._assemble_context({"role": "system", "content": "s"}, [_m("user", "now"), _m("assistant", "ok")])
        recalled = next(row for row in rows if row["content"] == "recalled memory")
        assert recalled["message_uid"] == engine_uid(
            lineage, "recall", hashlib.sha256(b"recalled memory").hexdigest(), 0)
        assert [row for row in rows if row["content"] in {"now", "ok"} and "message_uid" in row] == []
    finally:
        engine.shutdown()


# -- record -------------------------------------------------------------------------------------------------

def test_engine_rows_recorded_once_and_ignored_by_lookups_gate_and_doctor(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        out = _compress(engine, _host())
        uid, lineage = out[0]["message_uid"], _lineage(engine)
        assert _engine_rows(engine) == [(0, uid, "engine", "carrier")]
        _compress(engine, _host())
        assert len(_engine_rows(engine)) == 1
        store = engine._store
        assert store.host_uid_bindings_for(lineage, {uid}) == {}
        assert store.host_uid_bindings_for(lineage, {uid}, ("engine",)) == {uid: [(0, "engine")]}
        assert store.host_uid_uids_of_stores(lineage, {0}) == {}
        store.record_host_uid_checks(lineage, [(uid, 0, False)])
        assert store._conn.execute("SELECT first_check FROM host_uid_bindings WHERE kind = 'engine'").fetchone() \
            == (None,)
        before = store.host_uid_gate()
        store._conn.execute("UPDATE host_uid_bindings SET first_check = 'disagree', disagree_seen = 1 "
                            "WHERE kind = 'engine'")
        store._conn.commit()
        assert store.host_uid_gate() == before
        doctor = host_uid_doctor_lines(engine)
        assert "host_uid_gate_disagree: 0" in doctor and "host_uid_engine_rows: 1" in doctor
        assert f"host_uid_bindings_rows: {store.count_host_uid_bindings()}" in doctor  # F2: engine rows not counted
        assert store.count_host_uid_bindings() == store._conn.execute(
            "SELECT COUNT(*) FROM host_uid_bindings WHERE kind != 'engine'").fetchone()[0]
        # #836: a session delete purges bindings by stored row; store_id 0 rows stay, which is acceptable: they
        # name no row, are keyed by lineage and only ever feed GENERATED events.
        store.delete_session_messages("S")
        assert len(_engine_rows(engine)) == 1
    finally:
        engine.shutdown()


# -- descriptors ----------------------------------------------------------------------------------------------

def test_descriptor_engine_uid_is_optional(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        out = _compress(engine, _host())
        proof = engine._compress_commit_proof
        descriptor = proof["emissions"][0]
        assert descriptor["kind"] == "carrier" and descriptor["engine_uid"] == out[0]["message_uid"]
        legacy = {**proof, "emissions": [{k: v for k, v in d.items() if k != "engine_uid"}
                                         for d in proof["emissions"]]}
        assert all(_descriptor_shape_is_well_formed(d) for d in proof["emissions"] + legacy["emissions"])
        assert _project_emitted_occurrences(out, proof=proof) == _project_emitted_occurrences(out, proof=legacy)
        # A v0.25.0-shaped persisted proof (no field) still validates through the durable reader.
        payload = engine._durable_commit_proof_payload()
        assert payload["emissions"] and payload["emissions"][0]["engine_uid"] == out[0]["message_uid"]
        key = engine._replay_snapshot_metadata_key(_COMPACTION_COMMIT_PROOF_METADATA_PREFIX, "S")
        stored = engine._store.read_metadata_json(key)
        for item in stored["emissions"]:
            item.pop("engine_uid", None)
        engine._store.write_metadata_json([key], json.dumps(stored, sort_keys=True))
        reread = engine._durable_commit_proof_payload()
        assert reread["emissions"] and "engine_uid" not in reread["emissions"][0]
        assert [{k: v for k, v in d.items() if k != "engine_uid"} for d in payload["emissions"]] \
            == reread["emissions"]
    finally:
        engine.shutdown()


# -- carriers (sites 2, 19, 18) ------------------------------------------------------------------------------

def test_site2_carrier_key_set_and_ordered_absorbed(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    host = _host()
    host[-4]["_absorbed_message_uids"] = ["x", "u6", "y"]
    host[-4]["_row_id"] = 7
    try:
        carrier = _compress(engine, host)[0]
        assert set(carrier) == {"role", "content", "message_uid", "_absorbed_message_uids"}
        assert carrier["_absorbed_message_uids"] == ["u6", "x", "y"]
    finally:
        engine.shutdown()


def _cut(engine, out, budget, absorbed):
    """The survival cut over a compress output whose head is the carrier; captures the split remainder."""
    out = copy.deepcopy(out)
    if absorbed is not None:
        out[0]["_absorbed_message_uids"] = absorbed
    out[0]["_row_id"] = 3
    seen = []
    original = engine._survival_generated
    engine._survival_generated = lambda message: seen.append(message) or original(message)
    store_ids = engine._get_store_id_map_for_messages(out)
    fitted = engine._survival_cut(out, 1, budget, True, "test", store_ids, True)[0]
    return fitted, next(m for m in seen if m["content"].startswith("question"))


@pytest.mark.parametrize("absorbed", [["u4"], ["u4", "other"]])
def test_site19_reformed_carrier_and_site18_split(tmp_path, absorbed):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path, tail=8)
    try:
        out = _compress(engine, _host())
        uid = out[0]["message_uid"]
        fitted, remainder = _cut(engine, out, 10_000, absorbed)
        carrier = fitted[0]
        assert carrier["content"].endswith("question 5" + PAD)
        assert set(carrier) == {"role", "content", "message_uid", "_absorbed_message_uids"}
        assert (carrier["message_uid"], carrier["_absorbed_message_uids"]) == (uid, ["u5"])
        expected = {"role", "content"} | ({"message_uid"} if len(absorbed) == 1 else set())
        assert set(remainder) == expected
        if len(absorbed) == 1:
            assert remainder["message_uid"] == "u4"
    finally:
        engine.shutdown()


def test_site18_summary_only_row_keeps_the_engine_uid(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        out = _compress(engine, _host())
        fitted, remainder = _cut(engine, out, 10_000, None)
        summary_row = fitted[0]
        assert set(summary_row) == {"role", "content", "message_uid"}
        assert summary_row["message_uid"] == out[0]["message_uid"]
        assert set(remainder) == {"role", "content", "message_uid"} and remainder["message_uid"] == "u6"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["not_capable", "off"])
def test_survival_sites_are_base_shaped_without_the_gate(tmp_path, monkeypatch, case):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path, tail=8)
    try:
        out = _compress(engine, _host())
        out[0]["message_uid"] = "host-composite"  # a host-persisted carrier
        if case == "off":
            monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
        else:
            monkeypatch.setattr(emit, "_host_uid_capability", False)
        fitted, remainder = _cut(engine, out, 10_000, ["u4"])
        assert set(fitted[0]) == {"role", "content"}
        assert remainder["message_uid"] == "host-composite" and remainder["_row_id"] == 3
    finally:
        engine.shutdown()


# -- GENERATED (R3-3 class 2) ------------------------------------------------------------------------------

def test_restart_replay_counts_generated_agree_for_a_summary_and_a_carrier(tmp_path):
    _state_db(tmp_path, [("S", None, None), ("T", None, None)])
    counts = {}
    for session, tail in (("S", 4), ("T", 3)):  # S emits a carrier, T a summary row
        first = _open(tmp_path, session=session, tail=tail)
        try:
            out = _compress(first, _host())
            assert ("_absorbed_message_uids" in out[0]) is (session == "S")
        finally:
            first.shutdown()
        persisted = [dict(row, message_uid=row.get("message_uid") or f"minted-{i}") for i, row in enumerate(out)]
        second = _open(tmp_path, session=session, tail=tail)
        try:
            second._ingest_messages(persisted)
            counts[session] = _counts(second)
            assert "host_uid_gate_disagree: 0" in host_uid_doctor_lines(second)
        finally:
            second.shutdown()
    assert counts["S"].get("generated.agree.remainder") == 1
    assert counts["T"].get("generated.agree") == 1
    for got in counts.values():
        assert "replay.skipped.unmapped_replay" not in got
        assert not any("disagree" in key for key in got), got


@pytest.mark.parametrize("stored, remainder, rows, bound, expected", [
    (None, False, [], {}, "generated.agree"),
    (None, False, [], {"h": [(5, "canonical")]}, "generated.agree.remainder"),
    (9, False, [], {}, "generated.disagree.stored"),
    (9, True, [], {"h": [(9, "canonical")]}, "generated.agree.remainder"),
    (9, True, [], {"h": [(5, "canonical")]}, "generated.disagree.remainder"),
    (None, False, [{"store_id": 5}], {"h": [(5, "version")]}, "generated.agree.remainder"),
    (None, False, [{"store_id": 5}], {}, "generated.disagree.mapped_extra"),  # F4(a): an unbound host maps nothing
    (None, False, [{"store_id": 5}, {"store_id": 9}], {"h": [(5, "canonical")]}, "generated.disagree.mapped_extra"),
    (9, True, [{"store_id": 4}], {"h": [(9, "canonical")]}, "generated.disagree.mapped_extra"),  # a stored generated row
])
def test_generated_outcomes(stored, remainder, rows, bound, expected):
    from hermes_lcm.host_uid import HostUidShadowMixin
    assert HostUidShadowMixin._host_uid_generated("e", ["h"], {"e"}, bound, rows, stored, remainder) == expected
    if not bound and stored is None:
        assert HostUidShadowMixin._host_uid_generated("e", [], {"e"}, {}, rows, stored, remainder) == (
            "generated.agree" if not rows else "generated.disagree.mapped_extra")


# -- fix round 1 ---------------------------------------------------------------------------------------------

def test_f1_carrier_uid_is_the_summary_uid_and_the_descriptor_engine_uid(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        carrier = _compress(engine, _host())[0]
        descriptor = engine._compress_commit_proof["emissions"][0]
        assert descriptor["kind"] == "carrier" and descriptor["engine_uid"] == carrier["message_uid"]
        assert descriptor["generated_span_sha256"] != hashlib.sha256(
            carrier["content"][:engine._verified_lcm_summary_prefix_end(carrier["content"])].encode()).hexdigest()
        no_carrier = engine._assemble_context(None, copy.deepcopy(_host()[-2:]))  # one user row: a summary row
        assert "_absorbed_message_uids" not in no_carrier[0]
        assert no_carrier[0]["message_uid"] == carrier["message_uid"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("absorbed, expected", [(["e2"], None), (["u4", "e2"], "u4"), ([], None)])
def test_f3_site18_counts_host_uids_only(tmp_path, absorbed, expected):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path, tail=8)
    try:
        out = _compress(engine, _host())
        engine._store.add_host_uid_bindings(_lineage(engine), [(0, "e2", "engine", "recall")])  # durable only
        _fitted, remainder = _cut(engine, out, 10_000, absorbed)
        assert remainder.get("message_uid") == expected
        assert set(remainder) == {"role", "content"} | ({"message_uid"} if expected else set())
    finally:
        engine.shutdown()


def test_f5_identical_generated_rows_get_ordinals_and_reproduce(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        def mint():
            rows = [{"role": "user", "content": "same"}, {"role": "user", "content": "same"}]
            engine._mint_engine_uids([(row, "recall", None, "recall") for row in rows])
            return [row["message_uid"] for row in rows]
        first = mint()
        assert first[0] != first[1] and mint() == first
        basis = hashlib.sha256(b"same").hexdigest()
        assert first == [engine_uid(_lineage(engine), "recall", basis, n) for n in (0, 1)]
    finally:
        engine.shutdown()


def test_f5_absorbed_engine_uid_classes_generated_after_a_host_fold(tmp_path):
    _state_db(tmp_path, [("T", None, None)])
    first = _open(tmp_path, session="T", tail=3)
    try:
        out = _compress(first, _host())
    finally:
        first.shutdown()
    summary_uid = out[0]["message_uid"]
    persisted = [dict(row, message_uid=row.get("message_uid") or f"minted-{i}") for i, row in enumerate(out)]
    persisted[0] = {**persisted[0], "message_uid": "host-fold", "_absorbed_message_uids": [summary_uid]}
    second = _open(tmp_path, session="T", tail=3)
    try:
        second._ingest_messages(persisted)
        counts = _counts(second)
        assert counts.get("generated.agree") == 1, counts
        assert not any(key.startswith(("unbound", "replay.unbound", "replay.skipped")) for key in counts), counts
    finally:
        second.shutdown()


@pytest.mark.parametrize("capable_host", [True, False])
def test_f5_overflow_placeholder_minted_only_when_capable(tmp_path, monkeypatch, capable_host):
    from hermes_lcm.engine import _OVERFLOW_RECOVERY_PLACEHOLDER
    monkeypatch.setattr(emit, "_host_uid_capability", capable_host)
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        out = engine._assemble_overflow_recovery_context(
            None, [{"role": "tool", "tool_call_id": "orphan", "content": "status"}], assembly_cap_override=120)
        assert out[-1]["content"] == _OVERFLOW_RECOVERY_PLACEHOLDER
        if capable_host:
            basis = hashlib.sha256(_OVERFLOW_RECOVERY_PLACEHOLDER.encode()).hexdigest()
            assert out[-1]["message_uid"] == engine_uid(_lineage(engine), "overflow_placeholder", basis, 0)
        else:
            assert set(out[-1]) == {"role", "content"}
    finally:
        engine.shutdown()
