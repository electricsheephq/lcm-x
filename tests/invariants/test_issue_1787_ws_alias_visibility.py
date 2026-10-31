"""I1/I2: R1-ws alias diagnostics see first sightings without changing replay."""
from copy import deepcopy
import logging
import sqlite3

import pytest

from tests.invariants.test_issue_1519_alias_uid_conflict import deterministic_host, message
from tests.invariants.test_issue_1780_round3 import context
from tests.invariants.test_issue_1780_round6 import replay

__all__ = ["deterministic_host"]


def seed(path, uid="uid-a"):
    (path / "home").mkdir(parents=True)
    with sqlite3.connect(path / "home/state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")
    initial = [message("Fixture alias", 1000., **({"message_uid": uid} if uid else {})),
               message("Fixture anchor", 1001., message_uid="uid-o")]
    with context(path) as eng:
        eng.ingest(deepcopy(initial))
        eng._store.add_message_relations([[(1, "alt_stamp", None, None, 99.)]])
    return initial


def direct(path, caplog, text, uid, seed_uid="uid-a"):
    seed(path, seed_uid)
    view = [message(text, 99., **({"message_uid": uid} if uid else {}))]
    caplog.clear()
    with context(path) as eng:
        sql = []
        eng._store._conn.set_trace_callback(sql.append)
        try:
            with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
                plan = eng._identity_anchor_prematch(view, view, 0)
        finally:
            eng._store._conn.set_trace_callback(None)
    counts = [r.args for r in caplog.records if "would reject alias edges" in r.msg]
    reads = [s for s in sql if s.lstrip().upper().startswith("SELECT") and "json_each" in s]
    return plan, counts, reads


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_first_sighting_equals_plain_alias(tmp_path, monkeypatch, caplog, mode):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    plain, plain_counts, _ = direct(tmp_path / "plain", caplog, "Fixture alias", "uid-x")
    ws, ws_counts, reads = direct(tmp_path / "ws", caplog, "  Fixture alias\n", "uid-x")
    assert plain["matched"][0][0]["store_id"] == 1
    assert [r["store_id"] for r, _message in ws["ws"]] == [1]
    assert ws_counts == plain_counts == [(1,)]
    assert len(reads) == 2


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_each_cold_ingest_sees_whitespace_conflict(tmp_path, monkeypatch, caplog, mode):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    path = tmp_path / "arm"
    initial = seed(path)
    view = [message("  Fixture alias\n", 99., message_uid="uid-x"), initial[1],
            message("Fixture tail", 99., message_uid="uid-t")]
    counts = []
    for _ in range(3):
        caplog.clear()
        with context(path) as eng:
            with caplog.at_level(logging.DEBUG, logger="hermes_lcm.identity_anchor"):
                eng.ingest(deepcopy(view))
        counts.append(sum(r.args[0] for r in caplog.records if "would reject alias edges" in r.msg))
    assert counts[0] >= 1
    assert all(count >= 1 for count in counts)


@pytest.mark.parametrize("text", ["Fixture alias", "  Fixture alias\n"])
def test_off_has_no_observer_reads(tmp_path, monkeypatch, caplog, text):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    plan, counts, reads = direct(tmp_path / "arm", caplog, text, "uid-x")
    assert plan["replayed"] == {0}
    assert counts == reads == []


@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("uid,seed_uid", [("uid-a", "uid-a"), (None, "uid-a"), ("uid-x", None)],
                         ids=["same-uid", "no-incoming-uid", "missing-to-known"])
def test_nonconflicts_stay_quiet(tmp_path, monkeypatch, caplog, mode, uid, seed_uid):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    plan, counts, reads = direct(tmp_path / "arm", caplog, "  Fixture alias\n", uid, seed_uid)
    assert [r["store_id"] for r, _message in plan["ws"]] == [1]
    assert counts == []
    if uid is None:
        assert reads == []


@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("uid", ["uid-x", "uid-a"], ids=["conflict", "same-uid"])
def test_two_ingests_preserve_rows_bindings_and_ordered_writes(tmp_path, monkeypatch, caplog, mode, uid):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    arms = []
    for observe in (False, True):
        with monkeypatch.context() as patch:
            arms.append(replay(tmp_path / str(observe), patch, caplog, None, True, uid, observe))
    assert arms[0][:2] == arms[1][:2]
    assert not any("pre-match failed" in r.getMessage() for r in arms[1][2])


@pytest.mark.parametrize("mode", ["off", "shadow", "on"])
def test_matcher_keeps_original_keys_on_mixed_view(tmp_path, monkeypatch, mode):
    import hermes_lcm.identity_anchor as anchor

    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    path = tmp_path / "arm"
    seed(path)
    view = [message("  Fixture alias\n", 99., message_uid="uid-x"),
            message("Fixture alias", 99., message_uid="uid-a")]
    match = anchor._match_occurrences
    calls = []

    def original_keys_only(rows, keys_of, occurrences):
        keys = [keys_of(row) for row in rows]
        assert all((99., occurrences[0][1][1]) not in row_keys for row_keys in keys)
        assert any(occurrences[1][1] in row_keys for row_keys in keys)
        result = match(rows, keys_of, occurrences)
        calls.append(result)
        return result

    monkeypatch.setattr(anchor, "_match_occurrences", original_keys_only)
    plans = []
    for observe in (False, True):
        with context(path) as eng, monkeypatch.context() as patch:
            if not observe:
                patch.setattr(eng, "_observe_alias_uid_conflicts", lambda *a: None)
            plans.append(eng._identity_anchor_prematch(view, view, 0))
    assert calls == [{1: 1}, {1: 1}]
    assert plans[0] == plans[1]
    assert plans[1]["replayed"] == {1}
    assert 0 not in plans[1]["matched"]
