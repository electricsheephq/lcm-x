"""I1/I2: a failed count-only diagnostic must not discard the ingest plan."""
from copy import deepcopy
import json
import logging
import re
import sqlite3

import pytest

from tests.invariants.test_issue_1519_alias_uid_conflict import bindings, deterministic_host, message, rows
from tests.invariants.test_issue_1780_round3 import context

__all__ = ["deterministic_host"]


class DiagnosticFault:
    """Fail only the observer's indexed read; preserve every other connection operation."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, *args):
        if "INDEXED BY idx_host_uid_store" in sql:
            raise LookupError("private diagnostic fixture value")
        return self.conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def __enter__(self):
        return self.conn.__enter__()

    def __exit__(self, *args):
        return self.conn.__exit__(*args)


def normalized(value):
    # Only the per-arm home-derived lineage changes between otherwise identical fixtures.
    return re.sub(r"[0-9a-f]{16}:parent", "HOME:parent", json.dumps(value, sort_keys=True))


def replay(path, patch, caplog, fault, ws, uid, observe):
    from hermes_lcm.identity_anchor import IdentityAnchorMixin

    (path / "home").mkdir(parents=True)
    with sqlite3.connect(path / "home/state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")
    initial = [message("Fixture alias", 1000., message_uid="uid-a"),
               message("Fixture anchor", 1001., message_uid="uid-o")]
    view = [message("  Fixture alias\n" if ws else "Fixture alias", 99., message_uid=uid),
            initial[1], message("Fixture tail", 99., message_uid="uid-t")]
    with context(path) as eng:
        eng.ingest(deepcopy(initial))
        eng._store.add_message_relations([[(1, "alt_stamp", None, None, 99.)]])
    if not observe:
        patch.setattr(IdentityAnchorMixin, "_observe_alias_uid_conflicts", lambda *a: None)
    caplog.clear()
    snapshots, writes = [], []
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        for _ in range(2):
            with context(path) as eng:
                if fault == "drop_index":
                    with eng._store._conn:
                        eng._store._conn.execute("DROP INDEX IF EXISTS idx_host_uid_store")
                elif fault == "forced":
                    eng._store._conn = DiagnosticFault(eng._store._conn)
                sql = []
                eng._store._conn.set_trace_callback(sql.append)
                try:
                    eng.ingest(deepcopy(view))
                finally:
                    eng._store._conn.set_trace_callback(None)
                current = rows(eng)
                snapshots.append((current, [bindings(eng, r["store_id"]) for r in current]))
                writes.append([s for s in sql if not s.lstrip().upper().startswith(("SELECT", "PRAGMA", "WITH"))])
    logs = list(caplog.records)
    return normalized(snapshots), normalized(writes), logs


@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("fault,ws,uid", [
    pytest.param("surrogate", False, "uid-\ud800", id="surrogate-alias-conflict"),
    pytest.param("drop_index", False, "uid-x", id="drop-index-alias-conflict"),
    pytest.param("drop_index", True, "uid-x", id="drop-index-R1ws-alias-conflict"),
    pytest.param("drop_index", True, "uid-a", id="drop-index-R1ws-alias-same-uid"),
    pytest.param("forced", False, "uid-x", id="forced-observer-exception"),
])
def test_diagnostic_fault_preserves_ingest(tmp_path, monkeypatch, caplog, mode, fault, ws, uid):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    arms = []
    for observe in (False, True):
        with monkeypatch.context() as patch:
            arms.append(replay(tmp_path / str(observe), patch, caplog, fault, ws, uid, observe))
    assert arms[0][:2] == arms[1][:2], "diagnostic changed rows, bindings or ordered write SQL"
    assert not any("pre-match failed" in r.getMessage() for r in arms[1][2])
    # SQLite versions can consume an unmatched surrogate inside the IN subquery without decoding it.
    if fault == "surrogate":
        return
    failed = [r for r in arms[1][2] if "host UID observation unavailable" in r.msg]
    assert failed, "fixture did not exercise diagnostic fault containment"
    expected = "LookupError" if fault == "forced" else "OperationalError"
    assert all(r.levelno == logging.DEBUG and r.args == (expected,) for r in failed)
    assert all(r.getMessage() == f"LCM identity-anchor host UID observation unavailable ({expected})" for r in failed)
