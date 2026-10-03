"""v0.26.0 slice A: host ``message_uid`` SHADOW bindings, the R3-3 classifier and the doctor counts.

Shadow only records: #436's ingest decisions, the stored rows and the returned list are unchanged in every mode.
Engine-level (a host list in, rows / bindings / counters out) except the composite class, whose #436 outcome is
fed to the classifier directly."""

from __future__ import annotations

import json
import logging
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
    return conn.execute("SELECT store_id, uid, lineage_key, kind, binding_version, proof_kind FROM host_uid_bindings "
                        "ORDER BY rowid").fetchall()


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

    def boom(self):
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
        assert _counts(engine) == {"skipped.no_lineage_root": 1}
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
        return engine._host_uid_lineage_key()
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
        assert _counts(engine) == {"bound.agree.prefix_replay": 2, "unbound.agree_new": 1}
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
        assert _counts(engine) == {"skipped.unmapped_replay": 2}
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
        assert len(_bindings(engine)) == 1
    finally:
        engine.shutdown()


# -- anchored (#436) replays: AGREE_BIND, BOUND agree/disagree, ALIAS_CANDIDATE -------------------------

def _anchored(tmp_path, engine, seeded, host_view, *, bind=()):
    """Feed ``host_view`` to a restarted engine after seeding ``bind`` = [(content, uid, kind)]."""
    if bind:
        engine._store.add_host_uid_bindings("S", [(seeded[c], uid, kind, "seed") for c, uid, kind in bind])
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
        assert counts.get("unbound.agree_bind") == 3, counts
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
        assert counts.get("bound.agree.replay") == 1, counts
        assert counts.get("bound.disagree.replay_other_row") == 1, counts
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
        # with no bound neighbour after it in the view.
        _anchored(tmp_path, engine, seeded,
                  _restart_view(_m("user", "fresh" + PAD, 1.0),
                                [_m("user", "one" + PAD, 10.0, "u-1"), _m("assistant", "r1", 11.0, "u-r1-new"),
                                 _m("user", "two" + PAD, 12.0, "u-3"), _m("assistant", "r2", 13.0, "u-r2-new"),
                                 _m("user", "three" + PAD, 14.0)]),
                  bind=[("one" + PAD, "u-1", "canonical"), ("r1", "u-r1-old", "canonical"),
                        ("two" + PAD, "u-3", "canonical"), ("r2", "u-r2-old", "canonical"),
                        ("three" + PAD, "u-5", "canonical")])
        counts = _counts(engine)
        assert counts.get("unbound.alias_candidate.position_proof") == 1, counts
        assert counts.get("unbound.alias_candidate.unknown") == 1, counts
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
        assert _classify(engine, [merged, wrong], plan) == {"composite.agree.composite": 1,
                                                            "composite.disagree.composite": 1}
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
        assert "host_uid_counts: unbound.agree_new=3" in text
        assert "host_uid_process_counts: skipped.no_uid=1 unbound.agree_new=3" in text
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
        assert len(lines) == 1 and "agree=12" in lines[0] and "disagree=0" in lines[0]
        assert "u-1" not in lines[0] and "turn" not in lines[0]
    finally:
        engine.shutdown()
