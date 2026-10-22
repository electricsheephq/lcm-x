"""I1/I2: an unaudited changed prefix stays retryable after an anchor fault."""

from collections import Counter
from copy import deepcopy
import logging
import sqlite3

import pytest

from tests.test_host_uid_shadow import _engine, _m, _state_db


def _base():
    return [
        _m("user", "u one", 10.0),
        _m("assistant", "a one", 11.0),
        _m("user", "u two", 12.0),
        _m("assistant", "a two", 13.0),
        _m("user", "u three", 14.0),
        _m("assistant", "a three", 15.0),
    ]


def _boom(*_args, **_kwargs):
    raise sqlite3.OperationalError("synthetic prematch fault")


def _contents(engine):
    return [row["content"] for row in engine._store.get_session_messages("S")]


def _relations(engine):
    conn = engine._store._conn
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='message_relations'"
    ).fetchone():
        return []
    return [
        tuple(row)
        for row in conn.execute(
            "SELECT store_id, kind, related_store_id, ordinal, observed_at "
            "FROM message_relations ORDER BY relation_id"
        )
    ]


def _assert_once(engine, host, *, tail=False):
    contents = _contents(engine)
    assert contents.count(host[0]["content"]) == 1
    assert contents.count("u four") == int(tail)
    assert len(contents) == 7 + int(tail)
    assert len(contents) == len(set(contents))
    expected = [message["content"] for message in _base()] + [host[0]["content"]]
    assert Counter(contents) == Counter(expected + (["u four"] if tail else []))
    relations = _relations(engine)
    assert len(relations) == len(set(relations))


@pytest.mark.parametrize(
    "fault,rewrite",
    [
        (True, "content+stamp"),
        (False, "content+stamp"),
        (False, "stamp-only"),
        (False, "content-only-unstamped"),
        (True, "stamp-only"),
    ],
    ids=[
        "fault-content+stamp",
        "control-content+stamp",
        "control-stamp-only",
        "control-content-only-unstamped",
        "fault-stamp-only",
    ],
)
def test_rewritten_prefix_survives_prematch_fault(
    tmp_path, monkeypatch, caplog, rewrite, fault
):
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = _base()
        if rewrite == "content-only-unstamped":
            for message in host:
                message.pop("timestamp")
        e.ingest(deepcopy(host))
        initial = _contents(e)
        failures = e._ingest_failure_count
        host[0] = _m(
            "user",
            "u one" if rewrite == "stamp-only" else "u one REWRITTEN",
            None if rewrite == "content-only-unstamped" else 20.0,
        )
        real = e._identity_anchor_prematch
        with monkeypatch.context() as patch:
            if fault:
                patch.setattr(e, "_identity_anchor_prematch", _boom)
            with caplog.at_level(logging.WARNING):
                e.ingest(deepcopy(host))
        expected_failure = int(fault and rewrite != "stamp-only")
        observed_failures = e._ingest_failure_count
        assert e._consecutive_ingest_failures == 0
        assert not [
            record for record in caplog.records if record.levelno >= logging.ERROR
        ]
        assert e._identity_anchor_prematch == real
        e.ingest(deepcopy(host))
        e.ingest(deepcopy(host))
        if rewrite == "content+stamp":
            _assert_once(e, host)
        elif rewrite == "stamp-only":
            assert Counter(_contents(e)) == Counter(initial + ["u one"])
        else:
            # #633 audits inserted unstamped users, not content-only replacements.
            assert _contents(e) == initial
        assert observed_failures == failures + expected_failure
        assert e._last_active_replay_source_identities == [
            e._message_replay_identity(message, strip_carrier=False) for message in host
        ]
    finally:
        e.shutdown()


@pytest.mark.parametrize("faults", [1, 2])
@pytest.mark.parametrize("tail", [False, True], ids=["same-len", "tail"])
def test_rewrite_survives(tmp_path, monkeypatch, faults, tail):
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = _base()
        e.ingest(deepcopy(host))
        old_identity = e._last_active_replay_source_identities[0]
        host[0] = _m("user", "u one REWRITTEN", 20.0)
        if tail:
            host.append(_m("user", "u four", 21.0))
        failures = e._ingest_failure_count
        observations = []
        with monkeypatch.context() as patch:
            patch.setattr(e, "_identity_anchor_prematch", _boom)
            for count in range(1, faults + 1):
                e.ingest(deepcopy(host))
                observations.append(
                    (
                        e._ingest_failure_count,
                        e._last_active_replay_source_identities[0],
                    )
                )
                assert e._identity_anchor_inserted(host, 0, e._ingest_cursor) == set()
        e.ingest(deepcopy(host))
        e.ingest(deepcopy(host))
        _assert_once(e, host, tail=tail)
        assert observations == [
            (failures + count, old_identity) for count in range(1, faults + 1)
        ]
    finally:
        e.shutdown()


def test_fault_without_prefix_change_is_unchanged(tmp_path, monkeypatch):
    """An unchanged-prefix fault keeps the ordinary rows, cursor and exact cache."""
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = _base()
        e.ingest(deepcopy(host))
        rows = e._store.get_session_messages("S")
        failures = e._ingest_failure_count
        host.append(_m("user", "u four", 21.0))
        with monkeypatch.context() as patch:
            patch.setattr(e, "_identity_anchor_prematch", _boom)
            e.ingest(deepcopy(host))
            e.ingest(deepcopy(host))
        assert e._ingest_failure_count == failures
        assert e._store.get_session_messages("S")[:6] == rows
        assert _contents(e) == [message["content"] for message in host]
        assert e._ingest_cursor == len(host)
        assert e._last_active_replay_source_identities == [
            e._message_replay_identity(message, strip_carrier=False) for message in host
        ]
    finally:
        e.shutdown()


@pytest.mark.parametrize("stage", ["backfill", "version-rewind", "commit-relation"])
def test_partial_write_retry_is_idempotent(tmp_path, monkeypatch, stage):
    """Retry after real backfill, version rewind or a durable relation write."""
    _state_db(tmp_path, [("S", None, None)])
    e = _engine(tmp_path)
    try:
        host = _base()
        if stage == "backfill":
            host[2] = _m("user", "u two", uid="stable-two")
        elif stage == "commit-relation":
            host[4]["timestamp"] = 12.0
        e.ingest(deepcopy(host))
        host[0] = _m("user", "u one REWRITTEN", 20.0)
        if stage == "backfill":
            host[2]["timestamp"] = 12.0
        elif stage == "commit-relation":
            host = [
                host[0],
                host[1],
                _m("user", "u two\n\nu three", 12.0),
                host[3],
                host[5],
            ]
        failures = e._ingest_failure_count
        fired = []
        if stage == "version-rewind":
            owner, attribute = e, "_identity_anchor_version_rewind"
        else:
            owner = e._store
            attribute = (
                "backfill_observed_at_many"
                if stage == "backfill"
                else "add_message_relations"
            )
        real = getattr(owner, attribute)

        def after_write(*args, **kwargs):
            result = real(*args, **kwargs)
            if not fired and (stage == "version-rewind" or args[0]):
                if stage == "backfill":
                    assert e._store.get_session_messages("S")[2]["observed_at"] == 12.0
                elif stage == "commit-relation":
                    assert _relations(e)
                fired.append(True)
                raise sqlite3.OperationalError("synthetic fault after " + stage)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(owner, attribute, after_write)
            e.ingest(deepcopy(host))
        assert fired == [True], "fault must follow the real operation"
        observed_failures = e._ingest_failure_count
        relations_after_fault = _relations(e)
        e.ingest(deepcopy(host))
        e.ingest(deepcopy(host))
        _assert_once(e, host)
        assert _relations(e) == relations_after_fault
        assert observed_failures == failures + 1
        if stage == "commit-relation":
            assert len(relations_after_fault) == 2
            assert {relation[1] for relation in relations_after_fault} == {"composite"}
    finally:
        e.shutdown()
