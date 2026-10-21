"""I1/I2: alias UID-conflict observation preserves the existing ingest decisions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3
import time

import pytest

from tests.harness import engine_context, pin_char_counter


CASES = (
    "old_view_absent", "same_known_key", "different_known_key", "missing_to_known",
    "changed_insertion_row_id", "delayed_cold_replay_1505", "incoming_missing",
    "alias_candidate", "engine_incoming", "engine_stored", "other_lineage",
    "invalid_incoming", "invalid_stored", "non_alias",
)
TEXT = "Synthetic durable-key availability fixture. " * 500


def message(content, stamp, role="user", **keys):
    return {"role": role, "content": content, "timestamp": stamp, **keys}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rows(engine):
    return engine._store.get_session_messages("parent")


def targets(engine):
    return [r for r in rows(engine) if r["role"] == "user" and r["content"] == TEXT]


def bindings(engine, store_id):
    if not engine._store._host_uid_table_exists():
        return []
    return [tuple(r) for r in engine._store._conn.execute(
        "SELECT uid, kind, binding_version, proof_kind, first_check, disagree_seen "
        "FROM host_uid_bindings WHERE store_id=? ORDER BY rowid", (store_id,))]


@pytest.fixture(autouse=True)
def deterministic_host(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(time, "time", lambda: 1700000000.)
    home = tmp_path / "home"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")


def prepare(engine, case):
    """Adapt the six-case probe: real recovery, publication and host rewrite, never imported."""
    source = message(TEXT, 20., _row_id=101, platform_message_id="platform-old")
    if case not in {"old_view_absent", "missing_to_known"}:
        source["message_uid"] = "host-old"
    incoming = {**source, "timestamp": 30.}
    if case in {"old_view_absent", "different_known_key", "missing_to_known"}:
        incoming["platform_message_id"] = "platform-new"
        if case != "old_view_absent":
            incoming["message_uid"] = "host-new"
    if case == "changed_insertion_row_id":
        incoming["_row_id"] = 202
    if case in {"delayed_cold_replay_1505", "incoming_missing"}:
        incoming = message(TEXT, 30.)
    initial = [message("Synthetic earlier request", 10.), message("Synthetic earlier reply", 11., "assistant"),
               source, message("Synthetic orphan result", 21., "tool", tool_call_id="synthetic-orphan")]
    engine.ingest(initial)
    view = engine._assemble_overflow_recovery_context(None, initial, assembly_cap_override=120)
    view = engine._finalize_forced_overflow_result(initial, view, assembly_cap_override=120)
    engine._record_compress_commit_proof(initial, view)
    assert engine._last_compression_status == "overflow_recovery"
    assert any(e["kind"] == "recovery" for e in engine._compress_commit_proof["emissions"])
    view[-1]["timestamp"] = 30.
    engine.ingest(deepcopy(view))
    view[-1]["content"] = TEXT
    view.append(message("Synthetic recovery reply", 31., "assistant"))
    engine.ingest(view)
    assert len(targets(engine)) == 1
    sid = targets(engine)[0]["store_id"]
    assert len(engine._store.get_message_relations([sid], "alt_stamp")) == 1
    lineage, problem = engine._host_uid_lineage_key()
    assert lineage and problem is None
    # Narrow binding-kind controls; the alias itself was established through real ingest.
    if case in {"alias_candidate", "engine_stored", "other_lineage", "invalid_stored"}:
        column, value = {
            "alias_candidate": ("kind", "alias_candidate"),
            "engine_stored": ("kind", "engine"),
            "other_lineage": ("lineage_key", "another-home:root"),
            "invalid_stored": ("uid", "x" * 257),
        }[case]
        engine._store._conn.execute(f"UPDATE host_uid_bindings SET {column}=? WHERE store_id=?", (value, sid))
        engine._store._conn.commit()
        incoming["message_uid"] = "host-new"
    if case == "engine_incoming":
        from hermes_lcm.host_uid_emit import engine_uid

        uid = engine_uid(lineage, "recovery", "synthetic", 0)
        engine._store.add_host_uid_bindings(lineage, [(0, uid, "engine", "recovery")])
        incoming["message_uid"] = uid
    if case == "invalid_incoming":
        incoming["message_uid"] = 123
    if case == "non_alias":
        incoming.update(timestamp=20., message_uid="host-new")
    if case == "delayed_cold_replay_1505":
        view += [message("Synthetic later request", 40.), message("Synthetic later reply", 41., "assistant")]
        engine.ingest(view)
        return deepcopy(view)
    return [incoming, message("Synthetic new reply", 51., "assistant")]


@pytest.mark.parametrize("case", CASES)
def test_cold_alias_replay(tmp_path, case, record_property, caplog):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        replay = prepare(engine, case)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        old = targets(engine)[0]
        sid = old["store_id"]
        with caplog.at_level("DEBUG", logger="hermes_lcm.identity_anchor"):
            engine.ingest(replay)
        # Every row field, including the pinned clock fields, contributes to the ordered hash list.
        proof = {"ordered_row_hashes": [digest(r) for r in rows(engine)],
                 "bindings_hash": digest([bindings(engine, r["store_id"]) for r in rows(engine)])}
        record_property("byte_identity", json.dumps(proof, sort_keys=True))
        assert digest(next(r for r in rows(engine) if r["store_id"] == sid)) == digest(old)
        assert len(targets(engine)) == 1
        if case == "different_known_key":
            assert any("would reject" in r.message for r in caplog.records)



@pytest.mark.parametrize("conflict_first", [True, False])
@pytest.mark.parametrize("other_uid", [None, "host-old"])
def test_conflicting_edge_is_observed_without_changing_match(tmp_path, conflict_first, other_uid, caplog):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        prepare(engine, "same_known_key")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        old = targets(engine)[0]
        conflict = message(TEXT, 30., message_uid="host-new")
        other = message(TEXT, 30., **({"message_uid": other_uid} if other_uid else {}))
        view = [conflict, other] if conflict_first else [other, conflict]
        with caplog.at_level("DEBUG", logger="hermes_lcm.identity_anchor"):
            plan = engine._identity_anchor_prematch(view, view, 0)
        assert plan["replayed"] == {0}
        assert plan["matched"][0][0]["store_id"] == old["store_id"]
        assert any("would reject" in r.message for r in caplog.records)
        engine.ingest([*view, message("Synthetic final reply", 51., "assistant")])
        assert [r["observed_at"] for r in targets(engine)] == [20., 30.]
        assert digest(targets(engine)[0]) == digest(old)


@pytest.mark.parametrize("stamp,uid,same_text", [
    (30., None, True), (30., 123, True), (20., "host-new", True), (30., "host-new", False),
])
def test_no_binding_read_without_validated_alias_edge(tmp_path, monkeypatch, stamp, uid, same_text):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        prepare(engine, "same_known_key")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        def unexpected(*args, **kwargs):
            pytest.fail("prematch read bindings without a validated incoming alias edge")

        monkeypatch.setattr(engine._store, "host_uid_uids_of_stores", unexpected)
        monkeypatch.setattr(engine._store, "host_uid_bindings_for", unexpected)
        view = [message(TEXT if same_text else "Synthetic distinct occurrence", stamp, message_uid=uid)]
        sql = []
        engine._store._conn.set_trace_callback(sql.append)
        try:
            engine._identity_anchor_prematch(view, view, 0)
        finally:
            engine._store._conn.set_trace_callback(None)
        assert not [s for s in sql if s.lstrip().upper().startswith("SELECT") and "host_uid_bindings" in s]


def test_lookup_provenance_preserves_public_rows(tmp_path):
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        prepare(engine, "same_known_key")
        canonical = targets(engine)[0]
        projected = engine._store.find_rows_by_observed_at("conv", ["parent"], [30.])
        assert isinstance(projected, list)
        assert projected == [{**canonical, "observed_at": 30.}]
        assert projected.canonical_stamps == {canonical["store_id"]: 20.}
        assert engine._store.find_rows_by_observed_at("conv", ["parent"], []) == []
