"""SPEC-L2a-5 A-C/E: engine-built histories, cold classification, no wiring."""
from copy import deepcopy
import builtins
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from hermes_lcm import ingest_classifier as classifier, ingest_ledger as ledger
from hermes_lcm.engine import LCMEngine
from tests.harness import pin_char_counter
from tests.test_host_uid_shadow import _engine, _state_db
from tests.test_issue_2055_cold_cursor import emit, pure_summary


def msg(text, stamp=None, uid=None, role="user", **extra):
    return dict(role=role, content=text, **({"timestamp": stamp} if stamp is not None else {}),
                **({"message_uid": uid} if uid else {}), **extra)


def pair(stamp=None, uid=False, named=True):
    return [msg("", stamp, "ua" if uid else None, "assistant", tool_calls=[
        {"id": "c", "type": "function", "function": {"name": "read", "arguments": "{}"}}]),
        msg("ok", stamp + 1 if stamp is not None else None, "ut" if uid else None, "tool", tool_call_id="c",
            **({"tool_name": "read", "name": "read"} if named else {}))]


@pytest.fixture
def factory(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    monkeypatch.setattr(ledger, "_PENDING_EMISSIONS", {})
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        monkeypatch.setattr(LCMEngine, name, lambda *a: None)
    made = []

    def create(name="case", mode="off", *, cold_of=None, ledger_mode="shadow"):
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
        monkeypatch.setenv("LCM_INGEST_LEDGER", ledger_mode)
        root = tmp_path / name
        if cold_of is None:
            root.mkdir()
            _state_db(root, [("S", None, None)])
        else:
            cold_of.shutdown()
        engine = _engine(root)
        made.append(engine)
        return engine

    yield create
    for engine in reversed(made):
        engine.shutdown()


def snapshot(engine):
    conn = engine._store._conn
    lineage = ledger.persisted_lineage(conn, "S", ledger.lineage_key(engine._host_uid_lineage_key()[0], "S"))
    return classifier.SqliteLedgerSnapshot(conn, lineage, session_id="S")


def classify(engine, view, mode="off", *, snap=None):
    snap = snap or snapshot(engine)
    incoming = classifier.build_view(snap.salt, view, uid_mode=mode)
    expected, _ = engine._ledger_view_entries(snap.salt, view)
    assert [(r.identity_digest, r.stamp_digest, r.uidset_digest) for r in incoming] == [e[:3] for e in expected]
    return classifier.classify(incoming, snap, classifier.EngineOracle(engine), uid_mode=mode)


def materialize(engine, view, result):
    """Test-only L1 stand-in: apply the classifier's plan, never P1 decisions."""
    store, conn = engine._store, engine._store._conn
    snap = snapshot(engine)
    with store.atomic(immediate=True):
        sequence = ledger.next_view_seq(conn)
        pending = [i for i, v in enumerate(result.rows) if v.verdict == "new"]
        stored = dict(zip(pending, store.append_batch("S", [view[i] for i in pending], conversation_id="conv")))
        ordinals, respace = ledger.plan_ordinals(snap.ordinals(), [p for _, p in result.writes.placements])
        ledger.apply_respace(conn, snap.lineage, respace)
        bindings = [dict(ordinal=ordinal, store_id=stored[i], identity_digest=result.writes.entries[i].identity_digest,
                         stamp=ledger.stamp_text(view[i].get("timestamp")), disposition="stored", view_seq=sequence)
                    for (i, _), ordinal in zip(result.writes.placements, ordinals)]
        occurrences = dict(zip(pending, ledger.insert_bindings(conn, snap.lineage, bindings)))

        def resolve_store(ref):
            if isinstance(ref, classifier.NewRow):
                return stored[ref.index]
            return ref

        store.add_host_uid_bindings(snap.lineage, [(resolve_store(ref), uid, kind, proof)
                                   for ref, uid, kind, proof in result.writes.uid_bindings])
        for occ, stamp in result.writes.alt_stamps:
            ledger.add_stamp_observation(conn, snap.lineage, occ, stamp, "alt_stamp", sequence)
        for descriptor, stamp in result.writes.emission_stamps:
            conn.execute("UPDATE ledger_emissions SET observed_stamp=? WHERE lineage=? AND view_seq=? "
                         "AND output_index=? AND observed_stamp IS NULL", (stamp, snap.lineage, *descriptor))
        for seq in result.writes.adoptions:
            ledger.set_emission_state(conn, snap.lineage, seq, "adopted")
        if result.writes.stamp_unreliable:
            ledger._update_mark(conn, snap.lineage, "stamp_policy", "stamp_unreliable")
        for i in pending:
            relations = []
            for relation in result.rows[i].relations:
                target = relation.target
                related = snap.bindings[target]["store_id"] if isinstance(target, int) else None
                relations.append((stored[i], relation.kind, related, None, None))
            store.add_message_relations([relations] if relations else [])
        entries = []
        for entry in result.writes.entries:
            ref = entry.ref
            if isinstance(ref, classifier.NewRow):
                ref = occurrences[ref.index]
            if isinstance(ref, classifier.Descriptor):
                ref = next(e["emission_id"] for e in snap.emissions if
                           (e["view_seq"], e["output_index"]) == ref)
            entries.append(entry._replace(ref=ref))
        ledger.write_view(conn, snap.salt, snap.lineage, "ingest", None, entries, view_seq=sequence)
    return stored


def verdicts(result):
    return [v.verdict for v in result.rows]


def possible(verdict):
    return [r.target for r in verdict.relations if r.kind == "possible_replay_of"]


def history(factory, views, mode="off", name="case", *, cold_between=True):
    engine = factory(name, mode)
    for n, view in enumerate(views):
        if n and cold_between:
            engine = factory(name, mode, cold_of=engine)
        engine._ingest_messages(deepcopy(view))
    return factory(name, mode, cold_of=engine)


@pytest.mark.parametrize("mode", ["off", "shadow", "on"])
def test_a_purity_and_b_input_parity(factory, monkeypatch, mode):
    view = [msg("q", 1, "u"), *pair(2, True), msg("done", 4, "d", "assistant")]
    engine = history(factory, [view], mode)
    conn = engine._store._conn
    conn.execute("PRAGMA query_only=ON")
    before = conn.total_changes
    real_open, real_os_open, real_connect = builtins.open, os.open, sqlite3.connect

    def opening(path, mode="r", *args, **kwargs):
        assert not any(c in mode for c in "wax+"), (path, mode)
        return real_open(path, mode, *args, **kwargs)

    def os_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return real_os_open(path, flags, *args, **kwargs)

    def connect(path, *args, **kwargs):
        assert kwargs.get("uri") and "mode=ro" in str(path)
        return real_connect(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", opening)
        patch.setattr(os, "open", os_open)
        patch.setattr(sqlite3, "connect", connect)
        result = classify(engine, view, mode)
        assert verdicts(result) == ["replay"] * 4
        # Exercise aliases and all declared oracle read paths through the adapter.
        oracle, snap = classifier.EngineOracle(engine), snapshot(engine)
        for row in view:
            oracle.incoming_aliases(row, snap)
        for row in snap.bindings.values():
            oracle.stored_aliases(row, snap)
    assert conn.total_changes == before
    conn.execute("PRAGMA query_only=OFF")


def test_a_fresh_import_loads_no_engine():
    root = str(Path(__file__).resolve().parents[2])
    code = ("import sys,types,importlib; p=types.ModuleType('hermes_lcm'); "
            f"p.__path__=[{root!r}]; sys.modules['hermes_lcm']=p; "
            "importlib.import_module('hermes_lcm.ingest_classifier'); "
            "assert not any(n=='engine' or n.endswith('.engine') for n in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


@pytest.mark.parametrize("row", [msg("q"), pair()[0], pair()[1], dict(pair()[1], tool_name=None, name="read")])
def test_b_oracle_field_order(factory, row):
    engine = factory()
    row = dict(row, content="[Note: model was just switched from X to Y]\n\n" + row["content"])
    snap, oracle = snapshot(engine), classifier.EngineOracle(engine)
    raw = classifier.EngineOracle.identity_tuple(row)
    assert ledger.ledger_digest(snap.salt, *raw) == ledger.identity_digest(snap.salt, row)
    assert oracle.incoming_aliases(row, snap) == (raw,)


@pytest.mark.parametrize("text", [["A", "B", "C"], ["ok"] * 3])
def test_c_a1_fresh_restamp(factory, text):
    engine = history(factory, [[msg(t, i + 1) for i, t in enumerate(text)]])
    result = classify(engine, [msg(t, i + 7) for i, t in enumerate(text)])
    assert verdicts(result) == ["new"] * 3
    assert all(possible(v) and v.reason == "veto_stamp" for v in result.rows)
    assert not result.writes.stamp_unreliable


@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("anchored", [False, True])
def test_c_a3_revert(factory, mode, anchored):
    prefix = [msg("H", 1)] if anchored else []
    engine = history(factory, [[*prefix, msg("A", uid="u")], [*prefix, msg("B", uid="u")]], mode)
    result = classify(engine, [*prefix, msg("A", uid="u")], mode)
    assert result.rows[-1].verdict == "new"
    assert possible(result.rows[-1])
    assert any(r.kind == "supersedes" for r in result.rows[-1].relations) == anchored
    assert result.writes.uid_bindings[-1][2:] == ("version", "version_new")


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_c03_a3_original(factory, mode):
    engine = history(factory, [[msg("A", uid="u")], [msg("B", uid="u")], [msg("Z")]], mode)
    result = classify(engine, [msg("B", uid="u")], mode)
    assert verdicts(result) == ["replay"]
    assert result.rows[0].evidence == ("run_keyed" if mode == "on" else "run_unkeyed")


@pytest.mark.parametrize("mode", ["off", "shadow", "on"])
@pytest.mark.parametrize("witness", ["none", "stamp", "uid", "gw"])
def test_c_c4_name_elision(factory, mode, witness):
    stamped = witness == "stamp"
    keyed = witness in ("uid", "gw")
    original = [*pair(1 if stamped else None, keyed), msg("done", 3 if stamped else None, "ud" if keyed else None,
                                                       "assistant")]
    original[1]["content"] = "[Note: model was just switched from X to Y]\n\nok"
    engine = history(factory, [original], mode)
    view = deepcopy(original)
    view[1].update(content="ok")
    view[1].pop("tool_name")
    view[1].pop("name")
    view += [msg("ok", 4 if stamped else None, "ut2" if keyed else None, "tool", tool_call_id="c",
                 tool_name="read", name="read"), msg("next", 5 if witness in ("stamp", "gw") else None,
                                                   "un" if keyed else None)]
    result = classify(engine, view, mode)
    witnessed = stamped or (keyed and mode == "on")
    assert verdicts(result) == (["replay"] * 3 + ["new"] * 2 if witnessed else ["new"] * 5)
    assert classifier.build_view(snapshot(engine).salt, view, uid_mode=mode)[3].tool_segment is None
    if witnessed:
        assert not possible(result.rows[3])
    else:
        assert result.rows[0].reason == "segment" and result.rows[1].reason == "no_witness"
        assert result.rows[2].deferred == "rank5"
        if keyed and mode != "off":
            assert result.rows[3].reason == "veto_uid" and not result.rows[3].deferred
        else:
            assert result.rows[3].deferred == "rank5"


@pytest.mark.parametrize("named", [False, True], ids=["nameless", "name-only"])
def test_c_c4_externalized_stub_alias(factory, monkeypatch, named):
    from hermes_lcm.externalize import maybe_externalize_tool_output

    engine = factory()
    monkeypatch.setattr(engine._config, "large_output_externalization_enabled", True)
    content = "payload " * 100
    payload = maybe_externalize_tool_output(content, tool_call_id="c", session_id="S",
                                           config=engine._config, hermes_home=engine._hermes_home, force=True)
    assert payload is not None
    tool = msg(payload["placeholder"], role="tool", tool_call_id="c", **({"name": "read"} if named else {}))
    original = [msg("U"), pair()[0], tool, msg("done", role="assistant")]
    engine._ingest_messages(original)
    engine = factory(cold_of=engine)
    view = [*original[:2], msg(content, role="tool", tool_call_id="c"), original[3]]
    snap = snapshot(engine)
    binding = next(b for b in snap.bindings.values() if b["role"] == "tool")
    aliases = classifier.EngineOracle(engine).stored_aliases(binding, snap)
    expanded = next((t, needs) for t, needs in aliases if t[1] == content)
    assert expanded[1] == named
    result = classify(engine, view, snap=snap)
    if named:
        assert verdicts(result) == ["replay", "new", "new", "new"]
        assert result.rows[2].reason == "no_witness" and possible(result.rows[2]) == [binding["occ_id"]]
    else:
        assert verdicts(result) == ["replay", "replay", "replay", "new"]
        assert result.rows[3].deferred == "rank5"
        assert result.rows[2].evidence == "alias" and result.rows[2].target == binding["occ_id"]


def test_c_stamp_flip_revision_and_same_view_uid_writes(factory):
    engine = history(factory, [[msg("H", 1), msg("A", 2, "u")]], "shadow")
    flipped = [msg("H", 1), msg("A", 3, "u")]
    materialize(engine, flipped, classify(engine, flipped, "shadow"))
    result = classify(engine, [msg("H", 1), msg("A", 2, "u")], "shadow")
    assert result.rows[1].verdict == "new" and result.rows[1].reason == "revision"
    engine = factory("same-view", "shadow")
    result = classify(engine, [msg("A", uid="u"), msg("B", uid="u"), msg("C", uid="u")], "shadow")
    assert [u[2:] for u in result.writes.uid_bindings] == [
        ("canonical", "stored_new"), ("version", "version_new"), ("version", "version_new")]


@pytest.mark.parametrize("stamped", [False, True])
def test_c04_a5_swap(factory, stamped):
    p, q = msg("p", 1 if stamped else None), msg("q", 2 if stamped else None)
    engine = history(factory, [[p, q]])
    result = classify(engine, [q, p])
    assert verdicts(result) == ["replay", "new"]
    assert result.run.j == 1 and result.rows[1].reason == "window"
    assert possible(result.rows[1])


def test_c05_keyed_middle_and_c13_1449(factory):
    a, b, c = [msg(t, uid="u" + t) for t in "ABC"]
    engine = history(factory, [[a, b, c]], "on")
    result = classify(engine, [b, msg("M"), a], "on")
    assert verdicts(result) == ["replay", "new", "new"]
    assert result.rows[2].reason == "window"
    assert not any(r.kind == "supersedes" for r in result.rows[2].relations)
    view = [b, c, msg("D", uid="uA")]
    result = classify(engine, view, "on")
    assert verdicts(result) == ["replay", "replay", "new"]
    assert result.rows[2].reason == "window"
    materialize(engine, view, result)
    assert verdicts(classify(engine, view, "on")) == ["replay"] * 3


def test_c_crossed_and_clamped_placements(factory):
    h, a, b, c = msg("H"), *[msg(t, uid="u" + t) for t in "ABC"]
    engine = history(factory, [[h, a, b, c]], "on")
    snap = snapshot(engine)
    ords = snap.ordinals()
    for view in ([h, b, msg("X"), a], [h, c, msg("X"), a, msg("Y"), b]):
        result = classify(engine, view, "on")
        assert result.run.k == 1 and result.head == ("h1", ords[0])
        assert all(p.left == ords[2 if len(view) == 4 else 3] and p.right is None for _, p in result.writes.placements)
        new, changed = ledger.plan_ordinals(ords, [p for _, p in result.writes.placements])
        assert new == sorted(new) and not changed


def test_c01_h2_and_c02_h3(factory):
    original = [msg(t, i + 1) for i, t in enumerate(["U1", "A1", "U2", "A2"])]
    engine = history(factory, [original, [msg("W", 5)], [msg("Y", 6)]])
    result = classify(engine, original + [msg("U3", 7)])
    assert result.run is None and result.head.kind == "h2"
    assert verdicts(result) == ["replay"] * 4 + ["new"]
    result = classify(engine, [original[0], msg("Q", 8), msg("R", 9)])
    assert result.head.kind == "h4" and result.rows[0].reason == "head_h4"
    engine = history(factory, [[msg(t, uid=t) for t in "PQR"], [msg("W", uid="W")], [msg("Y", uid="Y")]], "on", "h3")
    result = classify(engine, [msg("P", uid="P"), msg("R", uid="R"), msg("N", uid="N")], "on")
    assert result.head.kind == "h3" and verdicts(result) == ["replay", "replay", "new"]
    result = classify(engine, [msg("R", uid="R"), msg("N", uid="N")], "on")
    assert result.head.kind == "h4" and result.counters["head.demoted"]


def test_c06_a6_second_row_and_c08_previous(factory):
    first = [msg("U1", 1), msg("A1", 2)]
    engine = history(factory, [first, [msg("C", 3)]])
    result = classify(engine, [msg("X", 0), *first, msg("U2", 4)])
    assert (result.run.i, result.run.j, result.run.k) == (1, 0, 2)
    assert result.previous.view_seq > result.run.record.view_seq
    assert verdicts(result) == ["new", "replay", "replay", "new"]
    assert result.writes.placements[0][1] == ledger.Placement(None, result.head.ordinal)


def test_c09_c9_emission_and_c07_merged_prefix(factory):
    engine = factory()
    original = [msg("U", 1), msg("A", 2, role="assistant")]
    engine._ingest_messages(original)
    summary = pure_summary(engine, "user")
    emit(engine, [summary, *original], {0: "summary"})
    snap = snapshot(engine)
    e = snap.emissions[0]
    descriptor = classifier.Descriptor(e["view_seq"], 0)
    result = classify(engine, [dict(summary, timestamp=5), msg("new", 6)])
    assert result.rows[0].verdict == "replay" and (descriptor, "5.0") in result.writes.emission_stamps
    with engine._store.atomic():
        engine._store._conn.execute("UPDATE ledger_emissions SET observed_stamp='5.0'")
    result = classify(engine, [dict(summary, timestamp=6)])
    assert result.rows[0].reason == "veto_stamp" and possible(result.rows[0]) == [descriptor]
    assert any(r.kind == "generated_unproven" for r in result.rows[0].relations)
    assert classify(engine, [dict(summary, timestamp=5)]).rows[0].verdict == "replay"
    # The cell specifies an adopted vector whose retained host rows have refs.
    # P1's initial emission writer leaves them unresolved; model the adopted
    # ledger state explicitly, without relaxing the classifier's ref requirement.
    with engine._store.atomic():
        record = next(r for r in snap.records if r["kind"] == "emission")
        entries = [entry._replace(ref=snap.by_digest[entry.identity_digest][0]["occ_id"])
                   if not entry.is_descriptor else entry for entry in record["entries"]]
        engine._store._conn.execute("UPDATE ingest_views SET entries=? WHERE kind='emission'",
                                   (ledger.encode_entries(entries),))
        ledger.set_emission_state(engine._store._conn, snap.lineage, e["view_seq"], "adopted")
    view = [dict(summary, content=summary["content"] + "\n\nU"), original[1], msg("U2", 3)]
    result = classify(engine, view)
    assert result.run.i == 1 and result.rows[0].deferred == "partial"
    assert len(possible(result.rows[0])) == 2
    suffix = classify(engine, [dict(summary, content="new prefix\n\n" + summary["content"])])
    assert suffix.rows[0].deferred == "partial" and possible(suffix.rows[0]) == [descriptor]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_c_m14_emission_threshold(factory, count):
    engine = factory()
    original = [msg(t, i + 1) for i, t in enumerate("UVW")]
    engine._ingest_messages(original)
    summaries = []
    for i in range(count):
        summary = pure_summary(engine, "user")
        summaries.append(summary)
        emit(engine, [*original[:i], summary], {i: "summary"})
    snap = snapshot(engine)
    with engine._store.atomic():
        for e in snap.emissions:
            engine._store._conn.execute("UPDATE ledger_emissions SET observed_stamp=? WHERE emission_id=?",
                                       (str(float(e["output_index"] + 1)), e["emission_id"]))
    result = classify(engine, [dict(s, timestamp=i + 10) for i, s in enumerate(summaries)])
    assert result.writes.stamp_unreliable == (count == 3)
    assert verdicts(result) == (["replay"] * count if count == 3 else ["new"] * count)
    if count < 3:
        assert all(v.reason == "veto_stamp" for v in result.rows)
    else:
        assert result.counters["m14.rerun"] == 1


@pytest.mark.parametrize("backfill", [False, True])
def test_c10_observed_at_veto_only(factory, backfill):
    original = [msg("U", 1), *pair(named=False)]
    engine = history(factory, [original])
    if backfill:
        with engine._store.atomic():
            engine._store._conn.execute("UPDATE messages SET observed_at=9 WHERE role='tool'")
    view = deepcopy(original)
    view[2]["timestamp"] = 5
    result = classify(engine, view)
    assert verdicts(result) == ["replay", "new", "new"]
    assert result.rows[1].reason == "segment"
    assert result.rows[2].reason == ("veto_stamp" if backfill else "deferred")
    assert result.rows[2].deferred == (None if backfill else "rank5")
    assert all(possible(v) for v in result.rows[1:])


def test_c10_user_backfill_never_witnesses(factory):
    engine = history(factory, [[msg("U", 1), msg("Y")]])
    with engine._store.atomic():
        engine._store._conn.execute("UPDATE messages SET observed_at=9 WHERE content='Y'")
    result = classify(engine, [msg("U", 1), msg("Y", 9)])
    assert result.rows[1].deferred == "rank5"


def test_c11_same_stamp_twins_uniqueness(factory):
    original = [msg("U", 1), msg("x", 2), msg("x", 2), msg("V", 3)]
    engine = history(factory, [original, [msg("W", 4)], [msg("Y", 5)]])
    result = classify(engine, original + [msg("N", 6)])
    assert verdicts(result) == ["replay"] * 4 + ["new"]
    assert result.rows[1].target < result.rows[2].target
    result = classify(engine, [original[0], original[1], original[3]])
    assert result.rows[1].verdict == "new" and result.rows[1].reason == "order"


@pytest.mark.parametrize("legacy", [False, True])
def test_c12_1519_and_c18_legacy_stamp(factory, legacy):
    original = [msg("X", 1), msg("A", 2), msg("B", 3)]
    engine = factory()
    engine._ingest_messages(original)
    snap = snapshot(engine)
    occurrence = next(iter(snap.by_digest[ledger.identity_digest(snap.salt, original[0])]))
    with engine._store.atomic():
        if legacy:
            engine._store.add_message_relations([[(occurrence["store_id"], "alt_stamp", None, None, 9.)]])
        else:
            ledger.add_stamp_observation(engine._store._conn, snap.lineage, occurrence["occ_id"], "9.0", "alt_stamp", 1)
    result = classify(engine, [original[1], original[2], msg("X", 9)])
    assert result.rows[2].reason == "window"
    result = classify(engine, [original[0], msg("A", 2), msg("B", 9)])
    assert result.rows[0].verdict == "replay"
    # A separate view witnesses the legacy observation at rank 2, after U's run.
    engine = factory("legacy")
    engine._ingest_messages([msg("U", 1), msg("X", 2)])
    snap = snapshot(engine)
    x = snap.by_digest[ledger.identity_digest(snap.salt, msg("X"))][0]
    with engine._store.atomic():
        engine._store.add_message_relations([[(x["store_id"], "alt_stamp", None, None, 9.)]])
    result = classify(engine, [msg("U", 1), msg("X", 9)])
    assert result.rows[1].evidence == "rank2" and not result.writes.alt_stamps


def test_c14_1967_segment_stamp_veto(factory):
    original = [msg("U", 1), *pair(2, named=False)]
    engine = history(factory, [original])
    result = classify(engine, [original[0], dict(original[1], timestamp=7), dict(original[2], timestamp=8)])
    assert verdicts(result) == ["replay", "new", "new"]
    assert [v.reason for v in result.rows[1:]] == ["veto_stamp"] * 2


def test_c15_duplicate(factory):
    engine = history(factory, [[msg("x"), msg("y")]])
    result = classify(engine, [msg("x"), msg("x"), msg("y")])
    assert verdicts(result) == ["replay", "new", "new"]
    assert not possible(result.rows[1]) and result.rows[2].deferred == "rank5"


def test_c16_evicted_history(factory):
    engine = history(factory, [[msg("A"), msg("B")]], name="evicted")
    first = classify(engine, [msg("A")])
    assert verdicts(first) == ["replay"] and first.rows[0].evidence == "run_unkeyed"
    assert first.rows[0].target == 1
    materialize(engine, [msg("A")], first)
    engine = factory("evicted", cold_of=engine)
    second = classify(engine, [msg("A"), msg("A")])
    assert verdicts(second) == ["replay", "new"]
    assert second.rows[0].target == 1 and second.rows[0].evidence == "run_unkeyed"
    assert second.rows[1].reason is None and not second.rows[1].relations
    materialize(engine, [msg("A"), msg("A")], second)
    engine = factory("evicted", cold_of=engine)
    snap = snapshot(engine)
    assert [(b["occ_id"], b["content"]) for b in sorted(snap.bindings.values(), key=lambda b: b["occ_id"])] == [
        (1, "A"), (2, "B"), (3, "A")]
    records = [r for r in reversed(snap.records) if r["kind"] == "ingest"]
    assert [[e.ref for e in r["entries"]] for r in records] == [[1], [1, 3]]
    result = classify(engine, [msg("A"), msg("B")], snap=snap)
    assert result.run == classifier.Run(classifier.Record("ingest", records[-1]["view_seq"]), 0, 0, 1)
    assert result.head == classifier.Head("h1", snap.bindings[1]["ordinal"])
    assert verdicts(result) == ["replay", "new"]
    assert result.rows[1].deferred == "rank5" and result.rows[1].reason == "deferred"
    assert possible(result.rows[1]) == [2]


def test_c20_anchored_e4_no_lineage_relation(factory):
    engine = history(factory, [[msg("A"), msg("B")], [msg("A")], [msg("A"), msg("A")]])
    snap = snapshot(engine)
    assert [(b["occ_id"], b["content"]) for b in sorted(snap.bindings.values(), key=lambda b: b["occ_id"])] == [
        (1, "A"), (2, "B"), (3, "A"), (4, "A"), (5, "A")]
    result = classify(engine, [msg("A"), msg("B")], snap=snap)
    assert result.run is not None and result.run.k == 1 and result.run.j == 0
    assert result.head == classifier.Head("h1", snap.bindings[4]["ordinal"])
    assert verdicts(result) == ["replay", "new"]
    assert result.rows[1].reason is None and result.rows[1].deferred is None
    assert result.rows[1].relations == []


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_c17_occurrence_proven_restamp(factory, mode):
    original = [msg(t, i + 1, "u" + t) for i, t in enumerate("ABC")]
    engine = history(factory, [original], mode)
    result = classify(engine, [dict(m, timestamp=i + 4) for i, m in enumerate(original)], mode)
    assert result.writes.stamp_unreliable == (mode == "on")
    assert verdicts(result) == (["replay"] * 3 if mode == "on" else ["new"] * 3)
    if mode == "on":
        assert result.counters["m14.rerun"] == 1 and len(result.writes.alt_stamps) == 3
    else:
        assert len(result.writes.uid_bindings) == 3


def test_c19_failed_emission_restart(factory, monkeypatch):
    engine = factory()
    original = [msg("U", 1), msg("A", 2, role="assistant")]
    engine._ingest_messages(original)
    summary = pure_summary(engine)
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "write_view", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("injected")))
        engine._compress_exit_write([summary, *original], ([summary, *original], [dict(kind="summary", output_occurrence={"index": 0, "of": 3})], None))
    engine = factory(cold_of=engine)
    result = classify(engine, [summary, *original])
    assert result.rows[0].verdict == "new"
    assert classifier.Relation("generated_unproven", None) in result.rows[0].relations


def test_e_cost(factory):
    """Nearest-rank percentiles, fresh snapshots, warm SQLite/OS page caches.

    Invocation is separate from correctness, so a >50 ms result is a stop signal.
    """
    engine = factory()
    original = [msg(f"row {i}", i + 1) for i in range(2000)]
    engine._ingest_messages(original)
    cases = {"exact": original}
    # Evict both vectors; three prefix rows corroborate h2 without a run.
    engine2 = factory("duplicates")
    duplicate_history = [msg(f"row {i % 20}", i + 1) for i in range(2000)]
    engine2._ingest_messages(duplicate_history)
    engine2._ingest_messages([msg("evict1", 3001)])
    engine2._ingest_messages([msg("evict2", 3002)])
    cases["no_run_duplicates"] = duplicate_history
    for name, view in cases.items():
        target = engine if name == "exact" else engine2
        salt = target._store.ingest_ledger_salt()
        incoming = classifier.build_view(salt, view, uid_mode="off")
        timings, pairs, row_pairs, queries = [], [], [], []
        for _ in range(20):
            statements = []
            target._store._conn.set_trace_callback(statements.append)
            started = time.perf_counter_ns()
            snap = snapshot(target)
            result = classifier.classify(incoming, snap, classifier.EngineOracle(target), uid_mode="off")
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
            target._store._conn.set_trace_callback(None)
            pairs.append(result.counters["candidate_pairs"])
            row_pairs.append(result.counters["row_candidate_pairs"])
            queries.append(len(statements))
            # snapshot(engine) also resolves the persisted lineage before its
            # ten snapshot reads; count that query in the measured path too.
            assert len(statements) == result.counters["queries"] + 1
            assert all(v.verdict == "replay" for v in result.rows)
            assert (result.run is not None) == (name == "exact")
        timings.sort()
        card = dict(case=name, samples=20, method="nearest-rank", cache="warm OS/SQLite, fresh snapshot",
                    p50_ms=timings[math.ceil(.50 * 20) - 1], p99_ms=timings[-1],
                    candidate_pairs=max(pairs), row_candidate_pairs=max(row_pairs), queries=max(queries),
                    timings_ms=timings)
        print("COST " + json.dumps(card, sort_keys=True))
        assert card["p99_ms"] <= 50, card
