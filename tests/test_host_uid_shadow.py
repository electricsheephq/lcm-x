"""v0.26.0 slice A: host ``message_uid`` SHADOW bindings, the R3-3 classifier and the doctor counts.

Shadow only records: #436's ingest decisions, the stored rows and the returned list are unchanged in every mode.
Engine-level (a host list in, rows / bindings / counters out) except the composite class, whose #436 outcome is
fed to the classifier directly."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import sqlite3

import pytest

import hermes_lcm.config as lcm_config
from hermes_lcm.command import _doctor_text
from hermes_lcm.config import LCMConfig, host_message_uid_mode
from hermes_lcm.db_bootstrap import SCHEMA_VERSION, get_schema_version, run_versioned_migrations
from hermes_lcm.engine import LCMEngine
from hermes_lcm.host_uid import HOST_UID_COUNTER_KEY
from hermes_lcm.store import MessageStore

PAD = " alpha beta gamma delta" * 4


def _m(role: str, text: str, ts: float | None = None, uid=None, **extra) -> dict:
    message = {"role": role, "content": text, **extra}
    if ts is not None:
        message["timestamp"] = ts
    if uid is not None:
        message["message_uid"] = uid
    return message


def _state_db(tmp_path, sessions) -> None:
    """Host state.db next to lcm.db: (id, parent_session_id, end_reason, source, model_config)."""
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                 "end_reason TEXT, source TEXT, model_config TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
                     [tuple(row) + (None,) * (5 - len(row)) for row in sessions])
    conn.commit()
    conn.close()


def _engine(tmp_path, session: str = "S", conversation: str = "conv") -> LCMEngine:
    config = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start(session, platform="cli", context_length=200_000, conversation_id=conversation)
    return engine


def _rows(engine: LCMEngine) -> list[tuple]:
    return [(r["store_id"], r["role"], r["content"], r.get("observed_at"), r.get("tool_call_id"))
            for r in engine._store.get_session_messages(engine._session_id)]


def _bindings(engine: LCMEngine) -> list[tuple]:
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='host_uid_bindings'").fetchone():
        return []
    return conn.execute("SELECT store_id, uid, substr(lineage_key, 18), kind, binding_version, proof_kind "  # root only
                        "FROM host_uid_bindings "
                        "ORDER BY rowid").fetchall()


def _root_of(key):
    """The root of a persisted ``<home_tag>:<root>`` lineage key (#836); None stays None."""
    return None if key is None else key.split(":", 1)[1]


def _root_and_problem(found):
    """``_host_uid_lineage_key()`` with the key reduced to its root, the problem kept: ``(root, problem)``."""
    key, problem = found
    return _root_of(key), problem


def _counts(engine: LCMEngine) -> dict:
    return {k: v for k, v in (getattr(engine, "_host_uid_counters", None) or {}).items() if v}


def _durable(engine: LCMEngine) -> dict:
    return engine._store.read_metadata_json(HOST_UID_COUNTER_KEY) or {}


def _id_of(engine: LCMEngine, content: str) -> int:
    return next(r[0] for r in _rows(engine) if r[2] == content)


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(lcm_config, "_host_message_uid_mode_warned", False)


# -- mode key -------------------------------------------------------------------------------------------

def test_mode_key_default_values_and_one_warning_for_invalid(monkeypatch, caplog):
    assert host_message_uid_mode() == "shadow"
    for value, expected in (("off", "off"), ("SHADOW", "shadow"), (" on ", "on"), ("", "shadow")):
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", value)
        assert host_message_uid_mode() == expected
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "bogus")
    with caplog.at_level(logging.WARNING):
        assert host_message_uid_mode() == "shadow"
        assert host_message_uid_mode() == "shadow"
    assert len([r for r in caplog.records if "LCM_HOST_MESSAGE_UID" in r.getMessage()]) == 1


# -- AGREE_NEW, durable tally, side table ---------------------------------------------------------------

def test_new_rows_bind_canonical_in_the_lineage_root(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "hello" + PAD, 10.0, "u-1"), _m("assistant", "hi", 11.0, "u-2")])
        user_id, reply_id = _id_of(engine, "hello" + PAD), _id_of(engine, "hi")
        assert _bindings(engine) == [(user_id, "u-1", "S", "canonical", 1, "stored_new"),
                                     (reply_id, "u-2", "S", "canonical", 1, "stored_new")]
        assert _counts(engine) == {"unbound.agree_new": 2}
        assert _durable(engine) == {"unbound.agree_new": 2}
    finally:
        engine.shutdown()


def test_on_mode_behaves_exactly_like_shadow(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "hello" + PAD, 10.0, "u-1")])
        assert [b[1:4] for b in _bindings(engine)] == [("u-1", "S", "canonical")]
        assert _counts(engine) == {"unbound.agree_new": 1}
    finally:
        engine.shutdown()


# -- off mode, no-uid host, fail-open -------------------------------------------------------------------

def _run(tmp_path, mode: str | None, monkeypatch, messages) -> tuple:
    if mode is None:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    tmp_path.mkdir(parents=True, exist_ok=True)
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        returned = engine._ingest_messages([dict(m) for m in messages])
        return (_rows(engine), returned, _bindings(engine), _counts(engine),
                engine._store.read_metadata_json(HOST_UID_COUNTER_KEY))
    finally:
        engine.shutdown()


_UID_LIST = [_m("user", "hello" + PAD, 10.0, "u-1"), _m("assistant", "hi", 11.0, "u-2"),
             _m("user", "again" + PAD, 12.0, "u-3")]


def test_off_mode_writes_and_classifies_nothing(tmp_path, monkeypatch):
    rows, _returned, bindings, counts, durable = _run(tmp_path, "off", monkeypatch, _UID_LIST)
    assert len(rows) == 3 and bindings == [] and counts == {} and durable is None
    conn = sqlite3.connect(tmp_path / "lcm.db")
    try:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='host_uid_bindings'").fetchone()
    finally:
        conn.close()


def test_no_uid_host_is_byte_identical_and_writes_no_table_row(tmp_path, monkeypatch):
    plain = [{k: v for k, v in m.items() if k != "message_uid"} for m in _UID_LIST]
    off = _run(tmp_path / "off", "off", monkeypatch, plain)
    shadow = _run(tmp_path / "shadow", None, monkeypatch, plain)
    assert shadow[0] == off[0] and shadow[1] == off[1]
    assert shadow[2] == [] and shadow[4] is None  # no binding, no durable tally
    assert shadow[3] == {"skipped.no_uid": 3}  # in memory only
    conn = sqlite3.connect(tmp_path / "shadow" / "lcm.db")
    try:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='host_uid_bindings'").fetchone()
    finally:
        conn.close()


def test_uid_host_store_result_and_returned_list_match_off_mode(tmp_path, monkeypatch):
    off = _run(tmp_path / "off", "off", monkeypatch, _UID_LIST)
    shadow = _run(tmp_path / "shadow", None, monkeypatch, _UID_LIST)
    assert shadow[0] == off[0] and shadow[1] == off[1]
    assert len(shadow[2]) == 3


def test_an_exception_in_shadow_code_leaves_ingest_unchanged(tmp_path, monkeypatch):
    off = _run(tmp_path / "off", "off", monkeypatch, _UID_LIST)

    def boom(self, *_args):
        raise RuntimeError("shadow failure")

    monkeypatch.setattr(LCMEngine, "_host_uid_lineage_key", boom)
    shadow = _run(tmp_path / "shadow", None, monkeypatch, _UID_LIST)
    assert shadow[0] == off[0] and shadow[1] == off[1] and shadow[2] == []
    assert shadow[3] == {"errors": 1} and shadow[4] == {"errors": 1}


def test_a_failing_capture_and_a_failing_table_write_fail_open(tmp_path, monkeypatch):
    off = _run(tmp_path / "off", "off", monkeypatch, _UID_LIST)

    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(MessageStore, "add_host_uid_bindings", boom)
    shadow = _run(tmp_path / "shadow", None, monkeypatch, _UID_LIST)
    assert shadow[0] == off[0] and shadow[1] == off[1]
    assert shadow[3].get("errors") == 1


# -- SKIPPED reasons ------------------------------------------------------------------------------------

def test_invalid_uids_are_skipped(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "a" + PAD, 10.0, 7), _m("user", "b" + PAD, 11.0, ""),
                       _m("user", "c" + PAD, 12.0, "x" * 257), _m("user", "d" + PAD, 13.0, "x" * 256)])
        assert _counts(engine) == {"skipped.invalid_uid": 3, "unbound.agree_new": 1}
        assert [b[1] for b in _bindings(engine)] == ["x" * 256]
    finally:
        engine.shutdown()


def test_no_state_db_gives_no_lineage_root(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "hello" + PAD, 10.0, "u-1")])
        assert _counts(engine) == {"skipped.no_lineage_root.unresolved": 1}  # a missing state.db is no error
        assert _bindings(engine) == []
    finally:
        engine.shutdown()


class _TimeoutRegex:  # the optional ``regex`` engine's timeout-capable interface (as tests/test_lcm_engine.py)
    error = re.error

    class _Pattern:
        def __init__(self, pattern):
            self._compiled = re.compile(pattern)

        def search(self, text, *, timeout=None):
            return self._compiled.search(text)

    @classmethod
    def compile(cls, pattern):
        return cls._Pattern(pattern)


def test_ignore_pattern_drop_is_skipped_not_stored(tmp_path, monkeypatch):
    from hermes_lcm import message_patterns

    monkeypatch.setattr(message_patterns, "_regex_engine", _TimeoutRegex)
    _state_db(tmp_path, [("S", None, None)])
    config = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"),
                       ignore_message_patterns=["^HEARTBEAT"])
    engine = LCMEngine(config=config)
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    try:
        engine.ingest([_m("user", "HEARTBEAT ok", 10.0, "u-1"), _m("user", "real" + PAD, 11.0, "u-2")])
        assert _counts(engine) == {"skipped.not_stored": 1, "unbound.agree_new": 1}
    finally:
        engine.shutdown()


# -- lineage root (R3-4) --------------------------------------------------------------------------------

def _root(tmp_path, session: str):
    engine = _engine(tmp_path, session=session)
    try:
        return _root_of(engine._host_uid_lineage_key()[0])
    finally:
        engine.shutdown()


def test_compression_child_shares_its_root_branch_and_reset_children_start_their_own(tmp_path):
    _state_db(tmp_path, [
        ("P", None, "compression"),
        ("C", "P", None),  # ordinary rotation
        ("B", "P", None, "cli", json.dumps({"_branched_from": "P"})),
        ("R", "P", None, "cli", json.dumps({"_reset_from": "P"})),
        ("D", "P", None, "cli", json.dumps({"_delegate_from": "P"})),
        ("T", "P", None, "tool"),
        ("X", "P", None, "cli", json.dumps({"_branched_from": "elsewhere"})),  # marker not bound to the parent
        ("Q", None, "user_exit"),
        ("Y", "Q", None),  # parent did not end by compression
    ])
    assert _root(tmp_path, "C") == "P"
    assert _root(tmp_path, "P") == "P"
    for child in ("B", "R", "D", "T", "Y"):
        assert _root(tmp_path, child) == child
    assert _root(tmp_path, "X") == "P"


def test_truncated_or_unreadable_chain_gives_no_root(tmp_path):
    chain = [("H0", None, "compression")] + [(f"H{i}", f"H{i - 1}", "compression") for i in range(1, 258)]
    _state_db(tmp_path, chain)
    assert _root(tmp_path, "H256") == "H0"  # exactly 256 hops: still read
    assert _root(tmp_path, "H257") is None  # over 256 hops
    assert _root(tmp_path, "missing") is None  # the session's own row is not readable
    (tmp_path / "state.db").write_bytes(b"not a database" * 100)
    assert _root(tmp_path, "H1") is None


def test_rotation_child_binds_under_the_parent_root(tmp_path):
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    engine = _engine(tmp_path, session="C")
    try:
        engine.ingest([_m("user", "hello" + PAD, 10.0, "u-1")])
        assert [b[1:4] for b in _bindings(engine)] == [("u-1", "P", "canonical")]
    finally:
        engine.shutdown()


# -- BOUND / UNBOUND outcomes on replay -----------------------------------------------------------------

def _seed(tmp_path, messages, session: str = "S"):
    engine = _engine(tmp_path, session=session)
    try:
        engine.ingest(messages)
        return {r[2]: r[0] for r in _rows(engine)}
    finally:
        engine.shutdown()


def test_restart_prefix_replay_of_a_bound_uid_agrees(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    first = [_m("user", "hello" + PAD, 10.0, "u-1"), _m("assistant", "hi", 11.0, "u-2")]
    _seed(tmp_path, first)
    engine = _engine(tmp_path)
    try:
        engine.ingest(first + [_m("user", "next" + PAD, 12.0, "u-3")])
        assert _counts(engine) == {"replay.bound.agree.prefix_replay": 2, "unbound.agree_new": 1}
        assert len(_bindings(engine)) == 3
    finally:
        engine.shutdown()


def test_restart_prefix_replay_of_an_unbound_uid_is_skipped_unmapped(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    first = [_m("user", "hello" + PAD, 10.0, "u-1"), _m("assistant", "hi", 11.0, "u-2")]
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    _seed(tmp_path, first)  # stored before shadow ran: no bindings
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID")
    engine = _engine(tmp_path)
    try:
        engine.ingest(first)
        assert _counts(engine) == {"replay.skipped.unmapped_replay": 2}
        assert _bindings(engine) == []
    finally:
        engine.shutdown()


def test_same_uid_new_bytes_is_version_new(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "hello" + PAD, 10.0, "u-1")])
        engine.ingest([_m("user", "hello" + PAD, 10.0, "u-1"), _m("user", "edited" + PAD, 10.5, "u-1")])
        edited = _id_of(engine, "edited" + PAD)
        assert _counts(engine) == {"unbound.agree_new": 1, "bound.version_new": 1}
        assert _bindings(engine)[-1] == (edited, "u-1", "S", "version", 1, "version_new")
    finally:
        engine.shutdown()


def test_same_uid_same_bytes_stored_again_disagrees(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "hello" + PAD, None, "u-1")])
        engine.ingest([_m("user", "hello" + PAD, None, "u-1"), _m("user", "hello" + PAD, None, "u-1")])
        assert _counts(engine) == {"unbound.agree_new": 1, "bound.disagree.stored_despite_match": 1}
        assert _gate(engine) == [("u-1", "disagree", 1)]  # the duplicate is a replay check on the matched binding
        assert "host_uid_gate_disagree: 1" in _doctor_text(engine)
        assert len(_bindings(engine)) == 1
    finally:
        engine.shutdown()


# -- anchored (#436) replays: AGREE_BIND, BOUND agree/disagree, ALIAS_CANDIDATE -------------------------

def _anchored(tmp_path, engine, seeded, host_view, *, bind=()):
    """Feed ``host_view`` to a restarted engine after seeding ``bind`` = [(content, uid, kind)]."""
    if bind:
        engine._store.add_host_uid_bindings(engine._host_uid_lineage_key()[0],
                                            [(seeded[c], uid, kind, "seed") for c, uid, kind in bind])
    engine.ingest(host_view)


def _restart_view(prefix_new, rows):
    """A restart whose first row is new (the ordered prefix proves nothing), then stamped stored rows."""
    return [prefix_new, *rows]


def test_anchored_replay_onto_an_unbound_row_binds_it(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    stored = [_m("user", "one" + PAD, 10.0), _m("assistant", "r1", 11.0), _m("user", "two" + PAD, 12.0)]
    seeded = _seed(tmp_path, stored)
    engine = _engine(tmp_path)
    try:
        view = _restart_view(_m("user", "fresh" + PAD, 1.0, "u-new"),
                             [_m("user", "one" + PAD, 10.0, "u-1"), _m("assistant", "r1", 11.0, "u-2"),
                              _m("user", "two" + PAD, 12.0, "u-3")])
        engine.ingest(view)
        counts = _counts(engine)
        assert counts.get("replay.unbound.agree_bind") == 3, counts
        bound = {b[1]: b[0] for b in _bindings(engine) if b[3] == "canonical"}
        assert bound["u-1"] == seeded["one" + PAD] and bound["u-3"] == seeded["two" + PAD]
    finally:
        engine.shutdown()


def test_anchored_replay_onto_the_bound_row_agrees_and_onto_another_row_disagrees(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    stored = [_m("user", "one" + PAD, 10.0), _m("assistant", "r1", 11.0), _m("user", "two" + PAD, 12.0)]
    seeded = _seed(tmp_path, stored)
    engine = _engine(tmp_path)
    try:
        _anchored(tmp_path, engine, seeded,
                  _restart_view(_m("user", "fresh" + PAD, 1.0),
                                [_m("user", "one" + PAD, 10.0, "u-1"), _m("assistant", "r1", 11.0),
                                 _m("user", "two" + PAD, 12.0, "u-3")]),
                  bind=[("one" + PAD, "u-1", "canonical"), ("r1", "u-3", "canonical")])
        counts = _counts(engine)
        assert counts.get("replay.bound.agree.replay") == 1, counts
        assert counts.get("replay.bound.disagree.replay_other_row") == 1, counts
    finally:
        engine.shutdown()


def test_alias_candidate_position_proof_versus_unknown(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    stored = [_m("user", "one" + PAD, 10.0), _m("assistant", "r1", 11.0), _m("user", "two" + PAD, 12.0),
              _m("assistant", "r2", 13.0), _m("user", "three" + PAD, 14.0)]
    seeded = _seed(tmp_path, stored)
    engine = _engine(tmp_path)
    try:
        # r1 is re-minted between two bound neighbours that map to its own stored neighbours; r2 is re-minted
        # after a neighbour whose view bytes are a NEW version, so the neighbour maps to another row.
        _anchored(tmp_path, engine, seeded,
                  _restart_view(_m("user", "fresh" + PAD, 1.0),
                                [_m("user", "one" + PAD, 10.0, "u-1"), _m("assistant", "r1", 11.0, "u-r1-new"),
                                 _m("user", "two" + PAD, 12.0, "u-3"), _m("assistant", "r2", 13.0, "u-r2-new"),
                                 _m("user", "three edited" + PAD, 14.5, "u-5")]),
                  bind=[("one" + PAD, "u-1", "canonical"), ("r1", "u-r1-old", "canonical"),
                        ("two" + PAD, "u-3", "canonical"), ("r2", "u-r2-old", "canonical"),
                        ("three" + PAD, "u-5", "canonical")])
        counts = _counts(engine)
        assert counts.get("replay.unbound.alias_candidate.position_proof") == 1, counts
        assert counts.get("replay.unbound.alias_candidate.unknown") == 1, counts
        aliases = [b for b in _bindings(engine) if b[3] == "alias_candidate"]
        assert sorted((b[0], b[1], b[5]) for b in aliases) == sorted([
            (seeded["r1"], "u-r1-new", "position_proof"), (seeded["r2"], "u-r2-new", "unknown")])
        # an alias candidate never becomes a binding
        assert not [b for b in _bindings(engine) if b[1] in ("u-r1-new", "u-r2-new") and b[3] != "alias_candidate"]
    finally:
        engine.shutdown()


# -- COMPOSITE / REMAINDER (classifier fed #436's outcome) ----------------------------------------------

def _classify(engine, messages, plan, stored_at=None, remainders=()):
    capture = engine._host_uid_capture(messages, messages, 0, 0, plan, set())
    engine._host_uid_shadow(capture, stored_at or {}, remainders)
    return _counts(engine)


def test_composite_and_remainder_classes(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    seeded = _seed(tmp_path, [_m("user", "r" + PAD, 10.0, "u-r"), _m("user", "u" + PAD, 11.0, "u-u"),
                              _m("user", "z" + PAD, 12.0, "u-z")])
    engine = _engine(tmp_path)
    try:
        rows = engine._store.get_batch([seeded["r" + PAD], seeded["u" + PAD]])
        group = [rows[seeded["r" + PAD]], rows[seeded["u" + PAD]]]
        merged = _m("user", "r" + PAD + "\n\n" + "u" + PAD, 10.0, "u-r", _absorbed_message_uids=["u-u"])
        wrong = _m("user", "r" + PAD + "\n\n" + "u" + PAD, 10.0, "u-r", _absorbed_message_uids=["u-z"])
        plan = {"replayed": {0, 1}, "matched": {0: group, 1: group}}
        assert _classify(engine, [merged, wrong], plan) == {"replay.composite.agree": 1,
                                                            "replay.composite.disagree": 1}
        before = len(_bindings(engine))
        remainder = _m("user", "r" + PAD + "\n\nnew tail", 10.0, "u-r")
        new_id = engine._store.append("S", {"role": "user", "content": "new tail"})
        counts = _classify(engine, [remainder], {"replayed": set(), "matched": {0: group[:1]}}, {0: new_id}, {0})
        assert counts.get("composite.agree.remainder") == 1
        assert len(_bindings(engine)) == before  # composites write no binding
    finally:
        engine.shutdown()


# -- old readers, doctor, compaction summary ------------------------------------------------------------

def test_old_reader_opens_a_store_that_has_the_table(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest(_UID_LIST)
        columns = [r[1] for r in engine._store._conn.execute("PRAGMA table_info(messages)")]
    finally:
        engine.shutdown()
    conn = sqlite3.connect(tmp_path / "lcm.db")
    try:
        run_versioned_migrations(conn)  # the version gate a v0.25.0 build runs: no refusal, no bump
        assert get_schema_version(conn) == SCHEMA_VERSION == 5
        assert [r[1] for r in conn.execute("PRAGMA table_info(messages)")] == columns
        assert conn.execute("SELECT 1 FROM lcm_migration_state WHERE step_name='host_uid_bindings_v1'").fetchone()
    finally:
        conn.close()
    store = MessageStore(str(tmp_path / "lcm.db"))
    try:
        assert [m["content"] for m in store.get_session_messages("S")] == [m["content"] for m in _UID_LIST]
    finally:
        store.close()


def test_doctor_reports_counts_only(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest(_UID_LIST + [_m("user", "plain" + PAD, 20.0)])
        text = _doctor_text(engine)
        assert "host_uid_mode: shadow" in text
        assert "host_uid_event_counts: unbound.agree_new=3" in text
        assert "host_uid_process_event_counts: skipped.no_uid=1 unbound.agree_new=3" in text
        assert "host_uid_gate_checked: 0" in text and "host_uid_gate_per_lineage: (none)" in text
        assert "host_uid_errors: 0" in text
        assert "host_uid_bindings_rows: 3" in text
        assert "u-1" not in text and "hello" not in text  # no uid, no content
    finally:
        engine.shutdown()


def test_one_info_summary_line_per_compaction(tmp_path, monkeypatch, caplog):
    import hermes_lcm.engine as lcm_engine

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **_k: ("Earlier turns.", 1))
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        live = []
        for i in range(6):
            live += [_m("user", f"turn {i}" + PAD * 20, 100.0 + i, f"u-{i}"), _m("assistant", f"r{i}", 100.5 + i, f"a-{i}")]
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(live, force=True)
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LCM host-uid shadow:")]
        assert len(lines) == 1 and "events agree=12" in lines[0] and "disagree=0" in lines[0]
        assert "u-1" not in lines[0] and "turn" not in lines[0]
    finally:
        engine.shutdown()


# -- fix round 1 ----------------------------------------------------------------------------------------

def _gate(engine: LCMEngine) -> list[tuple]:
    return engine._store._conn.execute(
        "SELECT uid, first_check, disagree_seen FROM host_uid_bindings WHERE kind != 'alias_candidate' ORDER BY rowid"
    ).fetchall()


def test_f1_restarts_replaying_one_prefix_count_each_binding_once(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    first = [_m("user", "hello" + PAD, 10.0, "u-1"), _m("assistant", "hi", 11.0, "u-2")]
    _seed(tmp_path, first)
    for _restart in range(3):
        engine = _engine(tmp_path)
        try:
            engine.ingest(first)
        finally:
            engine.shutdown()
    engine = _engine(tmp_path)
    try:
        assert _gate(engine) == [("u-1", "agree", 0), ("u-2", "agree", 0)]
        text = _doctor_text(engine)
        assert "host_uid_gate_checked: 2" in text and "host_uid_gate_agree: 2" in text
        assert "host_uid_gate_disagree: 0" in text and "host_uid_gate_per_lineage: 2/2/0" in text
        assert _durable(engine)["replay.bound.agree.prefix_replay"] == 6  # events still count every replay
    finally:
        engine.shutdown()


def test_f1_prefix_replay_disagreement_is_sticky_on_the_binding(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    _seed(tmp_path, [_m("user", "one" + PAD, 10.0, "u-a"), _m("user", "two" + PAD, 11.0, "u-b")])
    engine = _engine(tmp_path)
    try:  # the host now names the second stored row with the first row's uid
        engine.ingest([_m("user", "one" + PAD, 10.0, "u-a"), _m("user", "two" + PAD, 11.0, "u-a")])
        assert _counts(engine) == {"replay.bound.agree.prefix_replay": 1, "replay.bound.disagree.prefix_replay": 1}
        assert _gate(engine) == [("u-a", "agree", 1), ("u-b", None, 0)]
        assert "host_uid_gate_disagree: 1" in _doctor_text(engine)
    finally:
        engine.shutdown()


def test_f2_late_off_current_session_end_suffix_is_observed_under_that_session(tmp_path):
    _state_db(tmp_path, [("A", None, None), ("B", None, None)])
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path))
    engine.on_session_start("A", platform="cli", context_length=200_000)
    engine.on_session_start("B", platform="cli", context_length=200_000)
    try:
        engine.on_session_end("A", [_m("user", "late" + PAD, 10.0, "u-late"), _m("assistant", "ok", 11.0)])
        late = next(r for r in engine._store.get_session_messages("A") if r["content"] == "late" + PAD)
        assert [b[:4] for b in _bindings(engine)] == [(late["store_id"], "u-late", "A", "canonical")]
        assert _counts(engine) == {"unbound.agree_new": 1, "skipped.no_uid": 1}
    finally:
        engine.shutdown()


def test_f2_late_suffix_without_a_lineage_is_skipped(tmp_path):
    _state_db(tmp_path, [("B", None, None)])  # no row for A: unresolved
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path))
    engine.on_session_start("A", platform="cli", context_length=200_000)
    engine.on_session_start("B", platform="cli", context_length=200_000)
    try:
        engine.on_session_end("A", [_m("user", "late" + PAD, 10.0, "u-late")])
        assert len(engine._store.get_session_messages("A")) == 1
        assert _counts(engine) == {"skipped.no_lineage_root.unresolved": 1} and _bindings(engine) == []
    finally:
        engine.shutdown()


def test_f4_read_error_is_counted_and_unresolved_is_not(tmp_path):
    _state_db(tmp_path, [("S", None, None), ("C1", "C2", "compression"), ("C2", "C1", "compression")])
    assert _root(tmp_path, "C1") is None  # a cycle: unresolved
    engine = _engine(tmp_path, session="missing")
    try:
        assert engine._host_uid_lineage_key() == (None, "unresolved")
        engine.ingest([_m("user", "x" + PAD, 1.0, "u-x")])
        assert _counts(engine) == {"skipped.no_lineage_root.unresolved": 1}
    finally:
        engine.shutdown()
    (tmp_path / "state.db").write_bytes(b"not a database" * 100)
    engine = _engine(tmp_path)
    try:
        assert engine._host_uid_lineage_key() == (None, "read_error")
        engine.ingest([_m("user", "y" + PAD, 2.0, "u-y")])
        assert _counts(engine) == {"skipped.no_lineage_root.read_error": 1, "errors": 1}
    finally:
        engine.shutdown()


def test_f5_reviewer_counterexample_is_unknown_not_position_proof(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    seeded = _seed(tmp_path, [_m("user", "A1" + PAD, 10.0, "a"), _m("assistant", "X", 11.0, "x"),
                              _m("user", "B" + PAD, 12.0, "b")])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "A2" + PAD, 10.0, "a"), _m("assistant", "X", 11.0, "x-new"),
                       _m("user", "B" + PAD, 12.0, "b")])
        counts = _counts(engine)
        assert counts.get("replay.unbound.alias_candidate.unknown") == 1, counts
        assert "replay.unbound.alias_candidate.position_proof" not in counts
        assert [b[0] for b in _bindings(engine) if b[3] == "alias_candidate"] == [seeded["X"]]
    finally:
        engine.shutdown()


def test_f6_restart_classification_reads_rows_in_batches(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    view = [_m("user" if i % 2 == 0 else "assistant", f"row {i}", 100.0 + i, f"u-{i}") for i in range(5000)]
    _seed(tmp_path, view)
    calls = {"fetch": 0, "get_batch": 0}
    real_fetch, real_get_batch = LCMEngine._host_uid_fetch, MessageStore.get_batch

    def fetch(self, store_ids, fetched):
        calls["fetch"] += 1
        calls["inside"] = True
        try:
            return real_fetch(self, store_ids, fetched)
        finally:
            calls["inside"] = False

    def get_batch(self, store_ids):
        calls["get_batch"] += bool(calls.get("inside"))
        return real_get_batch(self, store_ids)

    monkeypatch.setattr(LCMEngine, "_host_uid_fetch", fetch)
    monkeypatch.setattr(MessageStore, "get_batch", get_batch)
    engine = _engine(tmp_path)
    try:
        engine.ingest(view)
        assert _counts(engine) == {"replay.bound.agree.prefix_replay": 5000}
        assert calls["fetch"] <= 2 and calls["get_batch"] <= 5000 // 500 + 2, calls
    finally:
        engine.shutdown()


_OLD_READER = r"""
import importlib.util, sqlite3, sys
old, db, root = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)  # the git-ignored agent/ stub
spec = importlib.util.spec_from_file_location("hermes_lcm", old + "/__init__.py", submodule_search_locations=[old])
sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
from hermes_lcm import store as store_mod
from hermes_lcm.command import _doctor_text
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
assert store_mod.__file__.startswith(old) and importlib.util.find_spec("hermes_lcm.host_uid") is None
engine = LCMEngine(config=LCMConfig(database_path=db))
engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
assert "sqlite_integrity: ok" in _doctor_text(engine)
before = [m["content"] for m in engine._store.get_session_messages("S")]
engine.ingest([{"role": "user", "content": "from the old build", "timestamp": 99.0, "message_uid": "u-old"}])
after = [m["content"] for m in engine._store.get_session_messages("S")]
assert after[:len(before)] == before and after[-1] == "from the old build", after
engine.shutdown()
print("OLD_READER_OK", len(before), len(after))
"""


# Old readers a rollback can land on. CI fetches both commits before pytest (#841).
ROLLBACK_READERS = {
    "v0.25.0-rc1": "c36b46e3a7b1bff23cfa45fa117565ab8624441d",
    "v0.25.1": "f47b55e031b507b424ff5f480d8f2a80d358f1f0",
}


def _skip_or_fail_in_ci(reason):
    # Under CI a missing prerequisite must not hide the rollback check (#841).
    if os.environ.get("CI"):
        pytest.fail(f"{reason}; CI must provide it")
    pytest.skip(reason)


@pytest.mark.parametrize("reader", sorted(ROLLBACK_READERS))
def test_f7_a_v0250_build_opens_ingests_and_reads_a_store_with_the_table(tmp_path, reader):
    import io
    import subprocess
    import sys
    import tarfile
    from pathlib import Path

    commit = ROLLBACK_READERS[reader]
    root = Path(__file__).resolve().parents[1]
    try:
        import agent.context_engine  # noqa: F401  (the git-ignored host stub the old build imports)
    except ImportError:
        _skip_or_fail_in_ci("the agent.context_engine host stub is not importable in this checkout")
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
        _skip_or_fail_in_ci(f"the {reader} commit {commit[:8]} is not in this checkout (shallow clone)")
    archive = subprocess.run(["git", "-C", str(root), "archive", commit], check=True, capture_output=True).stdout
    old = tmp_path / "old-reader"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter="data")
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest(_UID_LIST)
        assert len(_bindings(engine)) == 3
    finally:
        engine.shutdown()
    result = subprocess.run([sys.executable, "-c", _OLD_READER, str(old), str(tmp_path / "lcm.db"), str(root)],
                            capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OLD_READER_OK 3 4" in result.stdout


class _RaisingUidDict(dict):
    def get(self, key, default=None):
        if key == "message_uid":
            raise RuntimeError("host dict failure")
        return super().get(key, default)


def test_f7_capture_failure_leaves_ingest_byte_identical(tmp_path, monkeypatch):
    off = _run(tmp_path / "off", "off", monkeypatch, _UID_LIST)
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    (tmp_path / "shadow").mkdir()
    _state_db(tmp_path / "shadow", [("S", None, None)])
    engine = _engine(tmp_path / "shadow")
    try:
        returned = engine._ingest_messages([_RaisingUidDict(m) for m in _UID_LIST])
        assert _rows(engine) == off[0] and returned == off[1]
        assert _counts(engine) == {"errors": 1} and _bindings(engine) == []
    finally:
        engine.shutdown()


def test_f7_counter_flush_failure_leaves_ingest_unaffected(tmp_path, monkeypatch):
    off = _run(tmp_path / "off", "off", monkeypatch, _UID_LIST)

    def boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(MessageStore, "update_metadata_json", boom)
    shadow = _run(tmp_path / "shadow", None, monkeypatch, _UID_LIST)
    assert shadow[0] == off[0] and shadow[1] == off[1] and len(shadow[2]) == 3
    assert shadow[3] == {"unbound.agree_new": 3, "errors": 1}


def _all_rows(db) -> tuple:
    conn = sqlite3.connect(db)
    try:
        relations = conn.execute("SELECT * FROM message_relations ORDER BY relation_id").fetchall() if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone() else []
        return conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall(), relations
    finally:
        conn.close()


def _turns(start: int, count: int, base: float, uid: str) -> list[dict]:
    rows = []
    for i in range(start, start + count):
        rows += [_m("user", f"[T{i}] user turn {i}:" + PAD * 10, base + i * 10, f"{uid}-u{i}"),
                 _m("assistant", f"reply to T{i}", base + i * 10 + 1, f"{uid}-a{i}")]
    return rows


def _composite_steps():  # tests/test_issue_436_identity_anchor.py test_r2 (composite) and test_r3 (remainder)
    system = {"role": "system", "content": "stable system prompt"}
    head = [system, *_turns(1, 3, 0.0, "h")]
    r, u = _m("user", "R prompt" + PAD, 500.0, "u-r"), _m("user", "U prompt" + PAD, 510.0, "u-u")
    composite = _m("user", r["content"] + "\n\n" + u["content"], 500.0, "u-r", _absorbed_message_uids=["u-u"])
    yield "composite", [[*head, r, u], [*head, composite, _m("assistant", "reply to U", 511.0, "u-ru"),
                                        *_turns(10, 4, 600.0, "t")]]
    r = _m("user", "head R\n\n  indented line \n\n\nthree newlines\t", 500.0, "u-r")
    composite = _m("user", r["content"] + "\n\n  remainder U \n\nsecond  paragraph\n\n\n  tail \t", 500.0, "u-r")
    yield "remainder", [[*head[:5], r], [*head[:5], composite, _m("assistant", "reply", 501.0, "u-reply")]]


@pytest.mark.parametrize("case", ["composite", "remainder"])
def test_f7_composite_and_remainder_real_ingest_parity_with_off(tmp_path, monkeypatch, case):
    import time

    steps = dict(_composite_steps())[case]
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    results = {}
    for mode in ("off", "shadow"):
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
        (tmp_path / mode).mkdir()
        _state_db(tmp_path / mode, [("S", None, None)])
        engine = _engine(tmp_path / mode)
        try:
            for view in steps:
                engine.ingest([dict(m) for m in view])
            results[mode] = (_counts(engine), _all_rows(tmp_path / mode / "lcm.db"))
        finally:
            engine.shutdown()
    assert results["shadow"][1] == results["off"][1]
    assert results["off"][1][1], "the #436 relation must be recorded"
    expected = "replay.composite.agree" if case == "composite" else "composite.agree.remainder"
    assert results["shadow"][0].get(expected) == 1, results["shadow"][0]


def test_r2_version_new_leaves_bindings_unchecked_until_a_replay(tmp_path):
    """Reviewer A->B: a changed-bytes store is a VERSION_NEW event, never a gate check on A or B."""
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "A" + PAD, 10.0, "u")])
        engine.ingest([_m("user", "A" + PAD, 10.0, "u"), _m("user", "B" + PAD, 10.5, "u")])
        a_id, b_id = _id_of(engine, "A" + PAD), _id_of(engine, "B" + PAD)
        assert [(b[0], b[3]) for b in _bindings(engine)] == [(a_id, "canonical"), (b_id, "version")]
        assert _gate(engine) == [("u", None, 0), ("u", None, 0)]
        assert "host_uid_gate_checked: 0" in _doctor_text(engine)
        assert _counts(engine) == {"unbound.agree_new": 1, "bound.version_new": 1}
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:  # restart: the host replays u with A's bytes
        engine.ingest([_m("user", "A" + PAD, 10.0, "u")])
        assert _counts(engine) == {"replay.bound.agree.replay": 1}  # #436 maps it onto A
        assert _gate(engine) == [("u", "agree", 0), ("u", None, 0)]
        assert "host_uid_gate_checked: 1" in _doctor_text(engine)
    finally:
        engine.shutdown()



# -- fix round 3 (PR bot round 1) -----------------------------------------------------------------------

class _FaultyConn:
    """The store connection with one injected shadow-write failure: ``after_first_insert`` (a binding INSERT
    runs, then the batch fails) or ``commit`` (the shadow write fails at COMMIT)."""

    def __init__(self, real, fail):
        self._real, self._fail, self._armed = real, fail, False

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def executemany(self, sql, params):
        if "host_uid_bindings" in sql and self._fail:
            params = list(params)
            if self._fail == "after_first_insert" and sql.lstrip().startswith("INSERT"):
                self._fail = None
                self._real.execute(sql, params[0])
                raise sqlite3.OperationalError("injected after the first binding INSERT")
            self._armed = self._fail == "commit"
        return self._real.executemany(sql, params)

    def execute(self, sql, *args):
        if self._fail == "schema_commit" and "CREATE TABLE IF NOT EXISTS host_uid_bindings" in sql:
            self._armed = True
        return self._real.execute(sql, *args)

    def commit(self):
        if self._armed:
            self._armed = self._fail = None
            raise sqlite3.OperationalError("injected at COMMIT")
        return self._real.commit()


@pytest.mark.parametrize("fail", ["after_first_insert", "commit"])
def test_r3_a_failed_shadow_write_rolls_back_and_leaves_no_transaction(tmp_path, fail):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    store = engine._store
    real = store._conn
    try:
        store._conn = _FaultyConn(real, fail)
        engine.ingest(_UID_LIST)
        store._conn = real
        assert not real.in_transaction
        assert [r[2] for r in _rows(engine)] == [m["content"] for m in _UID_LIST]  # the messages were stored
        assert _bindings(engine) == []  # no partial binding
        assert _counts(engine).get("errors") == 1
        engine.ingest(_UID_LIST + [_m("user", "after" + PAD, 20.0, "u-after")])  # a normal ingest afterwards
        assert not real.in_transaction
        assert [b[1] for b in _bindings(engine)] == ["u-after"]
    finally:
        store._conn = real
        engine.shutdown()


def test_r3_the_lineage_cache_is_keyed_by_home(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    _state_db(tmp_path / "one", [("P", None, "compression"), ("S", "P", None)])
    _state_db(tmp_path / "two", [("S", None, None)])
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path / "one"))
    engine.on_session_start("S", platform="cli", context_length=200_000)
    try:
        assert _root_and_problem(engine._host_uid_lineage_key()) == ("P", None)
        engine._hermes_home = str(tmp_path / "two")  # the same session id under another profile home
        assert _root_and_problem(engine._host_uid_lineage_key()) == ("S", None)
    finally:
        engine.shutdown()


def test_r3_a_missing_state_db_is_read_once_per_session_and_is_no_error(tmp_path, monkeypatch):
    reads = []
    real = LCMEngine._host_uid_read_lineage

    def counted(self, path, session_id):
        reads.append(session_id)
        return real(self, path, session_id)

    monkeypatch.setattr(LCMEngine, "_host_uid_read_lineage", counted)
    engine = _engine(tmp_path)
    try:
        for i in range(3):
            engine.ingest([_m("user", f"turn {i}" + PAD, 10.0 + i, f"u-{i}") for i in range(i + 1)])
        assert reads == ["S"]
        assert _counts(engine) == {"skipped.no_lineage_root.unresolved": 3}
        assert "host_uid_errors: 0" in _doctor_text(engine)
    finally:
        engine.shutdown()


def test_r3_the_gate_counts_only_canonical_and_version_bindings(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m("user", "hello" + PAD, 10.0, "u-1")])
        store_id = _bindings(engine)[0][0]
        engine._store.add_host_uid_bindings(engine._host_uid_lineage_key()[0],
                                            [(store_id, "u-alias", "alias_candidate", "unknown")])
        engine._store._conn.execute("UPDATE host_uid_bindings SET first_check = 'disagree', disagree_seen = 1")
        engine._store._conn.commit()
        assert engine._store.host_uid_gate() == [(1, 0)]
        assert "host_uid_gate_checked: 1" in _doctor_text(engine)
    finally:
        engine.shutdown()



# -- fix round 4 ----------------------------------------------------------------------------------------

def _has_table(engine: LCMEngine) -> bool:
    return bool(engine._store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='host_uid_bindings'").fetchone())


@pytest.mark.parametrize("table", ["absent", "present"])
def test_r4_an_outer_transaction_is_never_committed_or_rolled_back(tmp_path, table):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    conn = engine._store._conn
    try:
        if table == "present":
            engine.ingest([_m("user", "first" + PAD, 1.0, "u-first")])
        stored = engine._store.append("S", {"role": "user", "content": "stored" + PAD})
        conn.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES ('S', 'user', 'outer pending', 1.0)")
        assert conn.in_transaction  # someone else's open transaction with a pending INSERT
        message = _m("user", "stored" + PAD, 5.0, "u-new")
        engine._host_uid_shadow(engine._host_uid_capture([message], [message], 0, 0, None, ()), {0: stored})
        assert conn.in_transaction  # still open, still owned by its holder
        assert _counts(engine)["errors"] == 2  # the skipped binding write and the skipped tally
        assert _has_table(engine) == (table == "present")
        conn.rollback()  # the holder decides: its pending row was never committed by the shadow
        assert not conn.execute("SELECT 1 FROM messages WHERE content = 'outer pending'").fetchone()
        assert "u-new" not in [b[1] for b in _bindings(engine)]
    finally:
        engine.shutdown()


def test_r4_a_schema_commit_failure_rolls_back_and_the_next_ingest_is_normal(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    store = engine._store
    real = store._conn
    try:
        store._conn = _FaultyConn(real, "schema_commit")
        engine.ingest(_UID_LIST)
        store._conn = real
        assert not real.in_transaction and not _has_table(engine)
        assert _counts(engine).get("errors") == 1
        assert len(_rows(engine)) == len(_UID_LIST)
        engine.ingest(_UID_LIST + [_m("user", "after" + PAD, 20.0, "u-after")])
        assert not real.in_transaction and [b[1] for b in _bindings(engine)] == ["u-after"]
    finally:
        store._conn = real
        engine.shutdown()


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores directory permissions")
def test_r4_a_permission_denied_state_db_is_a_read_error_not_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    _state_db(home, [("S", None, None)])
    real_exists = pathlib.Path.exists

    def exists_314(self, *args, **kwargs):  # Python 3.14: an OSError other than "not found" reads as False
        try:
            return real_exists(self, *args, **kwargs)
        except OSError:
            return False

    monkeypatch.setattr(pathlib.Path, "exists", exists_314)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(home))
    engine.on_session_start("S", platform="cli", context_length=200_000)
    try:
        home.chmod(0)  # the real path resolution (``_state_db_path``) now meets a permission-denied stat
        try:
            assert engine._host_uid_lineage_key() == (None, "read_error")
            assert engine._host_uid_lineage_key() == (None, "read_error")  # not cached as missing
            assert not engine.__dict__.get("_host_uid_lineage_cache")
        finally:
            home.chmod(0o700)
        assert _root_and_problem(engine._host_uid_lineage_key()) == ("S", None)
    finally:
        engine.shutdown()


def test_r4_the_lineage_cache_keeps_the_newest_512_entries(tmp_path):
    _state_db(tmp_path, [(f"s{i}", None, None) for i in range(600)])
    engine = _engine(tmp_path)
    try:
        for i in range(600):
            assert _root_and_problem(engine._host_uid_lineage_key(f"s{i}")) == (f"s{i}", None)
        cache = engine._host_uid_lineage_cache
        assert len(cache) == 512
        assert [key[1] for key in cache][:1] == ["s88"] and list(cache)[-1][1] == "s599"
    finally:
        engine.shutdown()
