"""#2055 P1b-2b: foreground shadow writes; comparison tree is Part A 783a8b824."""

import copy
import json
import logging
import re
import sqlite3
from pathlib import Path

import pytest

from hermes_lcm import config, ingest_ledger as ledger
from hermes_lcm.engine import LCMEngine
from hermes_lcm.sqlite_util import _SqliteSavepointError
from hermes_lcm.store import MessageStore
from tests.behaviour.reconcile_trace_harness import install_ledger_salt
from tests.harness import pin_char_counter
from tests.test_host_uid_shadow import _engine, _m, _state_db

SALT = b"l" * 32
TABLES = ("occurrence_bindings", "stamp_observations", "ingest_views", "ledger_emissions", "lineage_marks")


@pytest.fixture
def engines(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    install_ledger_salt(monkeypatch)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        monkeypatch.setattr(LCMEngine, name, lambda *args: None)
    made = []

    def create(name="case", *, mode="shadow", initialized=True, sessions=None):
        root = tmp_path / name
        root.mkdir()
        _state_db(root, sessions if sessions is not None else [("S", None, None)])
        monkeypatch.setenv("LCM_INGEST_LEDGER", mode)
        if initialized:  # Most tests isolate foreground behavior from the separate setup gate.
            store = MessageStore(root / "lcm.db")
            store.ensure_ingest_ledger()
            store.close()
        engine = _engine(root)
        made.append(engine)
        return engine

    yield create
    for engine in reversed(made):
        engine.shutdown()


def sql_rows(engine, table):
    return ledger._rows(engine._store._conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))


def count(engine, key):
    return getattr(engine, "_ingest_ledger_counters", {}).get("ledger." + key, 0)


def current_view(engine):
    return sql_rows(engine, "ingest_views")[-1]


def ledger_state(engine):
    return {table: sql_rows(engine, table) for table in (*TABLES, "ledger_meta")}


def restart(engine):
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_01_session_start_setup(engines, mode):
    engine = engines(mode=mode, initialized=False)
    assert engine._store.ingest_ledger_state() == ("ready" if mode == "shadow" else "absent")


def test_01_mode_parsing_and_setup_failure(engines, monkeypatch, caplog):
    monkeypatch.setattr(config, "_ingest_ledger_mode_warned", False)
    for value in ("", " shadow ", "on", "invalid"):
        monkeypatch.setenv("LCM_INGEST_LEDGER", value)
        assert config.ingest_ledger_mode() == "shadow"
    assert sum("invalid env LCM_INGEST_LEDGER" in r.getMessage() for r in caplog.records) == 1

    def fail(*args):
        raise sqlite3.OperationalError("injected setup failure")

    monkeypatch.setattr(MessageStore, "ensure_ingest_ledger", fail)
    engine = engines(initialized=False)
    restart(engine)
    assert count(engine, "setup_failed") == 2
    assert sum("ledger.setup_failed" in r.getMessage() for r in caplog.records) == 1
    engine._ingest_messages([_m("user", "still stored", 10.)])
    assert engine._store.get_session_messages("S")[0]["content"] == "still stored"


def test_02_bindings_use_raw_view_before_redaction(engines):
    engine = engines()
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["api_key"]
    view = [_m("user", "api_key=sk-user1234567890abcdef", "10", "user-1"),
            _m("assistant", "reply", 11., 17)]  # Invalid UID must not disable preparation.
    engine._ingest_messages(copy.deepcopy(view))
    stored, bindings = engine._store.get_session_messages("S"), sql_rows(engine, "occurrence_bindings")
    assert len(bindings) == len(stored) == 2
    assert stored[0]["content"] != view[0]["content"]
    assert [b["disposition"] for b in bindings] == ["stored", "stored"]
    assert [b["store_id"] for b in bindings] == [r["store_id"] for r in stored]
    assert bindings[0]["ordinal"] < bindings[1]["ordinal"]
    assert [b["stamp"] for b in bindings] == ["10.0", "11.0"]
    assert [b["identity_digest"] for b in bindings] == [ledger.identity_digest(SALT, row) for row in view]
    assert bindings[0]["identity_digest"] != ledger.identity_digest(SALT, stored[0])
    vector = current_view(engine)
    assert vector["state"] is None
    assert {b["view_seq"] for b in bindings} == {vector["view_seq"]}
    assert all(b["key_rank"] is None for b in bindings)
    entries = ledger.decode_entries(vector["entries"])
    assert [e.ref for e in entries] == [b["occ_id"] for b in bindings]
    assert entries[0].uidset_digest is not None and entries[1].uidset_digest is None
    assert not sql_rows(engine, "lineage_marks")
    assert engine._last_ingest_attribution["ledger_digest_ms"] >= 0
    assert engine._last_ingest_attribution["ledger_ms"] >= 0


def test_03_unchanged_view_adds_no_ledger_write_or_sequence(engines):
    engine = engines()
    view = [_m("user", "question", 10.), _m("assistant", "answer", 11.)]
    engine._ingest_messages(view)
    assert len(sql_rows(engine, "ingest_views")) == 1
    before, statements = ledger_state(engine), []
    engine._store._conn.set_trace_callback(statements.append)
    try:
        engine._ingest_messages(copy.deepcopy(view))
    finally:
        engine._store._conn.set_trace_callback(None)
    assert ledger_state(engine) == before
    # This no-uid, already-cached fixture is also write-free on Part A.
    assert not any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) for sql in statements)


def test_04_remainder_binding_uses_suffix_and_unbinds_whole_host_row(engines):
    engine = engines()
    head = [_m("user", "first", 1.), _m("assistant", "first reply", 2.)]
    donor = _m("user", "held donor\n\n indented", 10.)
    engine._ingest_messages([*head, donor])
    suffix = "  new suffix\n\nsecond paragraph\t"
    composite = {**donor, "content": donor["content"] + "\n\n" + suffix}
    engine._ingest_messages([*head, composite, _m("assistant", "reply", 11.)])
    row = next(row for row in engine._store.get_session_messages("S") if row["content"] == suffix)
    binding = next(b for b in sql_rows(engine, "occurrence_bindings") if b["store_id"] == row["store_id"])
    assert binding["disposition"] == "composite_part" and binding["stamp"] is None
    assert binding["identity_digest"] == ledger.identity_digest(SALT, {**composite, "content": suffix, "timestamp": None})
    entry = ledger.decode_entries(current_view(engine)["entries"])[2]
    assert entry.ref is None and entry.identity_digest == ledger.identity_digest(SALT, composite)


def _exercise_alt_stamp(engines, monkeypatch, bound, rollback):
    engine = engines(name=f"alt-{bound}-{rollback}", mode="shadow" if bound else "off")
    view = [_m("user", "seed", 10.), _m("assistant", "reply", 11.)]
    engine._ingest_messages(view)
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    row = engine._store.get_session_messages("S")[0]
    prematch, observed = engine._identity_anchor_prematch, []

    def alternate(*args, **kwargs):
        plan = prematch(*args, **kwargs)
        plan["relations"].append(("alt_stamp", 99., [row], None))
        return plan

    def after_relation(_pairs):
        assert engine._store.get_message_relations([row["store_id"]], "alt_stamp")
        observed.append(True)
        raise sqlite3.OperationalError("injected evidence I/O failure")

    monkeypatch.setattr(engine, "_identity_anchor_prematch", alternate)
    if rollback:
        monkeypatch.setattr(engine._store, "backfill_observed_at_many", after_relation)
    engine._ingest_messages(copy.deepcopy(view))
    observations = sql_rows(engine, "stamp_observations")
    assert len(observations) == int(bound and not rollback)
    if observations:
        assert observations[0]["kind"] == "alt_stamp" and observations[0]["stamp"] == "99.0"
        binding = next(b for b in sql_rows(engine, "occurrence_bindings") if b["store_id"] == row["store_id"])
        assert observations[0]["occ_id"] == binding["occ_id"]
    if rollback:
        assert observed == [True]
        assert not engine._store.get_message_relations([row["store_id"]], "alt_stamp")


def test_05_alt_stamp_is_bound_and_evidence_committed(engines, monkeypatch):
    for bound, rollback in ((True, False), (False, False), (True, True)):
        with monkeypatch.context() as patch:
            _exercise_alt_stamp(engines, patch, bound, rollback)


def test_06_carried_stamp_is_on_new_occurrence(engines):
    engine = engines()
    earlier, previous = _m("assistant", "earlier reply", 9.), _m("user", "interrupted", 10.)
    engine._ingest_messages([earlier, previous])
    engine._ingest_messages([earlier, _m("user", "replacement", 10.)])
    newest = engine._store.get_session_messages("S")[-1]
    binding = next(b for b in sql_rows(engine, "occurrence_bindings") if b["store_id"] == newest["store_id"])
    observations = sql_rows(engine, "stamp_observations")
    assert [(o["occ_id"], o["kind"], o["stamp"]) for o in observations] == [(binding["occ_id"], "carried_stamp", "10.0")]
    assert observations[0]["view_seq"] == binding["view_seq"]


@pytest.mark.parametrize("failure", ["integrity", "prepare", "busy", "savepoint"])
def test_07_fail_open_and_owner_propagation(engines, monkeypatch, caplog, failure):
    engine = engines()
    original = ledger.identity_digest if failure == "prepare" else ledger.insert_bindings
    calls, classifications = [], []
    classify = engine._capture_host_rewrites

    def classified(messages):
        classifications.append(True)
        return classify(messages)

    def failed(*args, **kwargs):
        calls.append(True)
        if failure == "busy":
            result = original(*args, **kwargs)
            if len(calls) == 1:
                error = sqlite3.OperationalError("injected busy snapshot")
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
                raise error
            return result
        if failure == "savepoint":
            raise _SqliteSavepointError("injected ledger savepoint failure")
        raise (ValueError if failure == "prepare" else sqlite3.IntegrityError)("injected ledger failure")

    monkeypatch.setattr(ledger, "identity_digest" if failure == "prepare" else "insert_bindings", failed)
    monkeypatch.setattr(engine, "_capture_host_rewrites", classified)
    view = [_m("user", "fresh", 10.)]
    if failure == "savepoint":
        with pytest.raises(_SqliteSavepointError, match="injected ledger savepoint failure"):
            engine._ingest_messages(view)
        assert not engine._store.get_session_messages("S")
        assert count(engine, "write_failed") == 0
    else:
        engine._ingest_messages(view)
        assert [r["content"] for r in engine._store.get_session_messages("S")] == ["fresh"]
        if failure == "busy":
            assert len(classifications) == len(calls) == 2
            assert len(sql_rows(engine, "occurrence_bindings")) == 1
            assert count(engine, "bindings_written") == count(engine, "views_written") == 1
        else:
            key = "prepare_failed" if failure == "prepare" else "write_failed"
            assert count(engine, key) == 1
            assert sum("ledger." + key in r.getMessage() for r in caplog.records) == 1
            assert all(not sql_rows(engine, table) for table in TABLES)
            assert engine._store._conn.execute("SELECT 1 FROM ledger_meta WHERE key LIKE 'session_lineage:%'").fetchone() is None
    assert calls and not engine._store._conn.in_transaction


def test_08_read_only_drain_and_delegate_exclusions(engines):
    """Lock-in: these paths leave existing ledger tables unchanged, even on Part A."""
    engine = engines()
    baseline = ledger_state(engine)
    locker = sqlite3.connect(engine._store.db_path)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", [_m("user", "frozen", 10.)])
        assert len(engine._pending_session_tails) == 1
        assert ledger_state(engine) == baseline
    finally:
        locker.rollback()
        locker.close()
    engine._retry_pending_session_tails()
    assert [r["content"] for r in engine._store.get_session_messages("S")] == ["frozen"]
    assert ledger_state(engine) == baseline
    assert engine._append_delegate_child_rows("child", [_m("user", "child row", 20.)],
        {"parent_session_id": "S", "source": "parent_frame", "child_subagent_id": "", "parent_subagent_id": "", "child_role": ""}) == 1
    assert engine._store.get_session_messages("child")[0]["content"] == "child row"
    assert ledger_state(engine) == baseline


@pytest.mark.parametrize("ancestry", ["resolved", "unresolved", "missing-parent", "later-unresolved", "drift"])
def test_09_lineage_persistence_and_legacy_resolution(engines, monkeypatch, ancestry):
    sessions = [] if ancestry == "unresolved" else [("S", "missing", None)] if ancestry == "missing-parent" else None
    engine = engines(sessions=sessions)
    legacy = engine._host_uid_lineage_key()[0]
    engine._ingest_messages([_m("user", "first", 10.)])
    conn = engine._store._conn
    persisted = conn.execute("SELECT value FROM ledger_meta WHERE key='session_lineage:S'").fetchone()
    if ancestry in {"unresolved", "missing-parent"}:
        assert persisted is None
        assert current_view(engine)["lineage"] == "provisional:S"
        assert engine._host_uid_lineage_key()[0] == legacy
        if ancestry == "missing-parent":
            assert legacy.endswith(":S")  # Legacy callers retain their short-root behavior and cache.
        return
    assert persisted[0] == legacy
    if ancestry in {"later-unresolved", "drift"}:
        lineage = engine._host_uid_lineage_key

        def changed(session_id=None, **kwargs):
            if kwargs.get("require_complete"):
                return (None, "unresolved") if ancestry == "later-unresolved" else ("other:root", None)
            return lineage(session_id)

        monkeypatch.setattr(engine, "_host_uid_lineage_key", changed)
        engine._ingest_messages([_m("user", "second", 20.)])
        assert conn.execute("SELECT value FROM ledger_meta WHERE key='session_lineage:S'").fetchone()[0] == legacy
        assert current_view(engine)["lineage"] == legacy
        assert count(engine, "lineage_drift") == int(ancestry == "drift")
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        engine._ingest_messages([_m("user", "third", 30.)])
    finally:
        conn.set_trace_callback(None)
    assert not any("INSERT INTO ledger_meta" in sql and "session_lineage:S" in sql for sql in statements)


def test_10_foreground_insert_respaces_exhausted_gap(engines):
    engine = engines()
    left, right = _m("user", "left", 10.), _m("assistant", "right", 20.)
    engine._ingest_messages([left, right])
    conn = engine._store._conn
    bindings = sql_rows(engine, "occurrence_bindings")
    assert len(bindings) == 2
    with engine._store.atomic():
        ledger.apply_respace(conn, bindings[0]["lineage"], {b["ordinal"]: i + 1 for i, b in enumerate(bindings)})
    restart(engine)
    engine._ingest_messages([left, _m("user", "middle", 15.), right])
    bindings = sql_rows(engine, "occurrence_bindings")
    by_content = {engine._store.get(b["store_id"])["content"]: b["ordinal"] for b in bindings}
    assert by_content["left"] < by_content["middle"] < by_content["right"]
    assert min(b - a for a, b in zip(sorted(by_content.values()), sorted(by_content.values())[1:])) >= ledger.MIN_GAP
    assert count(engine, "respaced") == 1


def test_11_off_sql_matches_pinned_part_a(engines, monkeypatch):
    """Lock-in: with the ledger off, an ingest's SQL equals Part A's (783a8b824) for the same fixture.

    The expected statements were captured once from the Part A engine, so the test needs no git history.
    """
    from tests.behaviour.reconcile_trace_harness import install_clock
    install_clock(monkeypatch)
    candidate = engines(mode="off", initialized=False)
    view = [_m("user", "same input", 10.), _m("assistant", "same answer", 11.)]
    statements = []
    candidate._store._conn.set_trace_callback(statements.append)
    try:
        candidate._ingest_messages(copy.deepcopy(view))
    finally:
        candidate._store._conn.set_trace_callback(None)
    expected = json.loads((Path(__file__).parent / "fixtures" / "ingest_ledger_off_sql_part_a.json").read_text())

    def product(sql):
        # Drop SQLite's own FTS5 statements, as the ingest trace harness does: they vary by SQLite version.
        return [s for s in sql if not s.lstrip().startswith("--") and not re.search(r"(['\"])(main|temp)\1\s*\.", s)]

    assert product(statements) == product(expected)


def test_12_absorbed_uid_change_updates_foreground_vector(engines):
    engine = engines()
    view = [_m("user", "unchanged", 10., "u", _absorbed_message_uids=["a", 7, "", "a"])]
    engine._ingest_messages(view)
    first = ledger.decode_entries(current_view(engine)["entries"])[0]
    view[0] = {**view[0], "_absorbed_message_uids": ("a", "b")}
    engine._ingest_messages(view)
    second = ledger.decode_entries(current_view(engine)["entries"])[0]
    assert first.uidset_digest == ledger.uidset_digest(SALT, "u", ["a"])
    assert second.uidset_digest == ledger.uidset_digest(SALT, "u", ["a", "b"])
    assert first.identity_digest == second.identity_digest and first.uidset_digest != second.uidset_digest
    assert len(sql_rows(engine, "occurrence_bindings")) == 1


def test_13_crossed_existing_bounds_skip_only_ledger(engines, monkeypatch):
    engine = engines()
    left, right = _m("user", "left", 10.), _m("assistant", "right", 20.)
    engine._ingest_messages([left, right])
    stored = engine._store.get_session_messages("S")
    prematch = engine._identity_anchor_prematch

    def crossed(*args, **kwargs):
        plan = prematch(*args, **kwargs)
        plan["cursor"] = 0
        plan["matched"] = {0: [stored[1]], 2: [stored[0]]}
        plan["replayed"] = {0, 2}
        return plan

    before = ledger_state(engine)
    monkeypatch.setattr(engine, "_identity_anchor_prematch", crossed)
    engine._ingest_messages([right, _m("user", "middle", 15.), left])
    assert [r["content"] for r in engine._store.get_session_messages("S")] == ["left", "right", "middle"]
    assert count(engine, "skipped_crossed") == 1
    assert ledger_state(engine) == before


def test_14_fixed_salt_and_write_view_optional_inputs(engines, monkeypatch):
    engine = engines()
    conn = engine._store._conn
    assert engine._store.ingest_ledger_salt() == SALT
    entries = [ledger.ViewEntry(b"d" * 16, None, None, None)]
    with engine._store.atomic():
        first = ledger.write_view(conn, SALT, "test", "ingest", None, entries)
        chain = ledger.prefix_chain(SALT, entries)
        seq = ledger.next_view_seq(conn)
        with monkeypatch.context() as patch:
            patch.setattr(ledger, "prefix_chain", lambda *args: pytest.fail("precomputed chain was recomputed"))
            assert ledger.write_view(conn, SALT, "test", "ingest", None, entries,
                                     view_seq=seq, precomputed_chain=chain) == seq
        assert first + 1 == seq
        assert ledger.newest_view(conn, "test", "ingest")["prefix_chain"] == chain
        assert ledger.decode_entries(ledger.newest_view(conn, "test", "ingest")["entries"]) == entries


def test_r1_read_only_sentinel_is_debug_and_preserves_frozen_tail(engines, monkeypatch, caplog):
    engine = engines()
    prefix = [_m("user", "seed", 10.), _m("assistant", "reply", 11.)]
    engine._ingest_messages(prefix)
    view = [*prefix, _m("user", "fresh", 20.)]
    prematch = engine._identity_anchor_prematch

    def replay_with_carry(*args, **kwargs):
        plan = prematch(*args, **kwargs)
        plan["carry"] = engine._store.get_session_messages("S")
        return plan

    monkeypatch.setattr(engine, "_identity_anchor_prematch", replay_with_carry)
    before = engine._ingest_failure_count
    locker = sqlite3.connect(engine._store.db_path)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with caplog.at_level(logging.DEBUG):
            engine.on_session_end("S", view)
        assert [row["content"] for row in engine._pending_session_tails[0]["messages"]] == ["fresh"]
        warnings = [r for r in caplog.records if "identity-anchor pre-match failed" in r.getMessage()]
        assert warnings and all(r.levelno == logging.DEBUG for r in warnings)
        assert engine._ingest_failure_count == before
    finally:
        locker.rollback()
        locker.close()


def test_r2_real_read_only_prematch_error_counts_once(engines, monkeypatch):
    """Passes pinned base 2af9177b4; fails Part A 783a8b824 (discarded callback)."""
    engine = engines()
    prefix = [_m("user", "seed", 10.), _m("assistant", "reply", 11.)]
    engine._ingest_messages(prefix)
    view = [{**prefix[0], "content": "changed seed"}, prefix[1], _m("user", "fresh", 20.)]
    errors = []

    def fail(_messages, _identities, _cursor, audit_from, **kwargs):
        assert getattr(engine, "_ingest_read_only", None) is not None and audit_from is not None
        errors.append(True)
        raise ValueError("real pre-match failure")

    monkeypatch.setattr(engine, "_identity_anchor_prematch", fail)
    before = engine._ingest_failure_count
    locker = sqlite3.connect(engine._store.db_path)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", view)
        assert errors == [True]
        assert engine._ingest_failure_count == before + 1
        assert engine._last_ingest_error == "ValueError: real pre-match failure"
        assert len(engine._pending_session_tails) == 1
        assert engine._pending_session_tails[0]["messages"][-1]["content"] == "fresh"
    finally:
        locker.rollback()
        locker.close()
