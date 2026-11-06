"""#2305: annotate cursor-0 copies without changing the stored message stream."""

from collections import Counter, defaultdict
from copy import deepcopy
import sqlite3

import pytest

from bench.instruments.reliability.scorers.multiset import split_surplus
from hermes_lcm.engine import LCMEngine
from tests.test_issue_1286_cleanup_rewrite_proof import (
    SID, Host, _change, _commit_cleanup, _turns,
    _no_summaries, _pin_lcm_char_counter,  # noqa: F401
)


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        monkeypatch.setattr(LCMEngine, name, lambda *args: None)
    instance = Host(tmp_path, embeddings_enabled=False)
    yield instance
    instance.close()


def _key(row):
    return row["role"], row.get("content"), row.get("tool_call_id"), row.get("tool_calls")


def _pairs(engine):
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone():
        return []
    return list(conn.execute(
        "SELECT store_id,related_store_id FROM message_relations "
        "WHERE kind='possible_replay_of' ORDER BY store_id,related_store_id"))


def _counts(engine):
    return getattr(engine, "_ingest_ledger_counters", Counter())


def _assert_cursor0(engine):
    decision = engine._last_ingest_reconciliation
    assert (decision["action"], decision["reason"], decision["cursor"]) == (
        "persisted batch", "persisted ambiguous delta", 0)


def _insert_case(host, *, cold=False):
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    before = host.rows()
    added = [{"role": "user", "content": "insert before cursor", "timestamp": host.stamp()},
             {"role": "user", "content": "next turn", "timestamp": host.stamp()},
             {"role": "assistant", "content": "next reply"}]
    view = deepcopy(host.live)
    view.insert(2, added[0])
    view += added[1:]
    if cold:
        host.restart()
    return before, view, added


def _assert_whole_store(host, before, view, added, surplus):
    # Main stores every unstamped row again, plus the genuinely new rows, in view order.
    expected_new = [_key(row) for row in view if not row.get("timestamp") or row in added]
    assert host.rows() == before + expected_new
    wanted = Counter(before + [_key(row) for row in added])
    rows = host.engine._store.get_session_messages(SID)
    ids = defaultdict(list)
    by_id = {row["store_id"]: row for row in rows}
    for row in rows:
        ids[_key(row)].append(row["store_id"])
    pairs = _pairs(host.engine)
    new_ids = {row["store_id"] for row in rows[len(before):]}
    surplus_ids = {sid for key, held in ids.items() for sid in held[wanted[key]:]}
    assert len(surplus_ids) == surplus
    assert Counter(source for source, _ in pairs) == Counter({sid: 1 for sid in surplus_ids})
    assert len({target for _, target in pairs}) == len(pairs)
    for source, target in pairs:
        assert source in new_ids and target < min(new_ids)
        assert by_id[source]["session_id"] == by_id[target]["session_id"] == SID
        assert _key(by_id[source]) == _key(by_id[target])
    groups = [(key, max(0, len(held) - wanted[key]), held) for key, held in ids.items()]
    split = split_surplus(groups, [(s, "possible_replay_of", t) for s, t in pairs])
    assert split["silent_surplus_rows"] == 0
    assert split["visible_surplus_rows"] == surplus
    assert _counts(host.engine)["ledger.replay_pairs.cursor0"] == surplus
    assert host.engine.get_status()["ingest_ledger"]["ledger.replay_pairs.cursor0"] == surplus


def test_01_mid_history_insert(host):
    before, view, added = _insert_case(host)
    host.engine.ingest(view)
    _assert_cursor0(host.engine)
    _assert_whole_store(host, before, view, added, 7)


def test_02_field_chain_replace_cold(host):
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    host.turn()
    host.turn()
    before = host.rows()
    changed = _change(host, "replace", insert_at=len(host.live) - 3, replace_at=len(host.live) - 4)
    added = [next(row for row in host.live if row["content"] == changed),
             {"role": "user", "content": "next turn", "timestamp": host.stamp()},
             {"role": "assistant", "content": "next reply"}]
    view = deepcopy(host.live) + added[1:]
    host.restart()
    host.engine.ingest(view)
    _assert_cursor0(host.engine)
    _assert_whole_store(host, before, view, added, 9)


def _ok_candidates(host, n=3):
    for _ in range(n):
        host.turn(reply="ok")
    return [row["store_id"] for row in host.engine._store.get_session_messages(SID)
            if row["role"] == "assistant"]


def _ingest_copies(host, copies):
    # A new first row cannot prove an ordered prefix. Copies are engine-ingested, never manually inserted.
    host.restart()
    before = host.stored()
    host.engine.ingest([{"role": "user", "content": "different head", "timestamp": host.stamp()}, *copies])
    _assert_cursor0(host.engine)
    return [row["store_id"] for row in host.engine._store.get_session_messages(SID)[before + 1:]]


def test_03_ties_use_oldest_targets(host):
    targets = _ok_candidates(host)
    sources = _ingest_copies(host, [{"role": "assistant", "content": "ok"}] * 2)
    assert len(sources) == 2
    assert _pairs(host.engine) == list(zip(sources, targets[:2]))
    assert _counts(host.engine)["ledger.replay_pairs.cursor0"] == 2


@pytest.mark.parametrize("copies", [1, 3])
def test_04_existing_target_is_not_reused(host, copies):
    targets = _ok_candidates(host)
    host.engine._store.add_message_relations([[(targets[1], "possible_replay_of", targets[0], None, None)]])
    sources = _ingest_copies(host, [{"role": "assistant", "content": "ok"}] * copies)
    assert _pairs(host.engine) == [(targets[1], targets[0]), *zip(sources, targets[1:])]
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == max(0, copies - 2)
    assert host.engine.get_status()["ingest_ledger"]["ledger.replay_pairs.cursor0_unlinked"] == max(0, copies - 2)


@pytest.mark.parametrize("stamp", [None, "equal", "different"])
def test_05_stamp_veto(host, monkeypatch, stamp):
    host.turn(reply="ok")
    known = host.stamp()
    host.turn(reply="stamped reply")
    # A known assistant host timestamp must be stored through the ordinary engine path.
    host.live.append({"role": "assistant", "content": "known stamp", "timestamp": known})
    host.engine.ingest(host.live)
    target = host.engine._store.get_session_messages(SID)[-1]["store_id"]
    copy = {"role": "assistant", "content": "known stamp"}
    if stamp is not None:
        copy["timestamp"] = known if stamp == "equal" else known + 100
    # Disable anchor consumption so the equal-stamp case exercises the relation writer.
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    sources = _ingest_copies(host, [copy])
    assert len(sources) == 1
    assert _pairs(host.engine) == ([] if stamp == "different" else [(sources[0], target)])
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == int(stamp == "different")


def test_06_cold_mid_history_insert(host):
    before, view, added = _insert_case(host, cold=True)
    host.engine.ingest(view)
    _assert_cursor0(host.engine)
    _assert_whole_store(host, before, view, added, 7)


def test_07_ordinary_append_does_not_link_even_after_cursor0(host):
    _ok_candidates(host)
    _ingest_copies(host, [{"role": "assistant", "content": "ok"}])
    before = _pairs(host.engine)
    view = [{"role": "user", "content": "different head", "timestamp": host.clock},
            {"role": "assistant", "content": "ok"}]
    host.engine.ingest([*view, {"role": "user", "content": "again", "timestamp": host.stamp()},
                        {"role": "assistant", "content": "ok"}])
    assert _pairs(host.engine) == before


def test_08_ledger_off(host, monkeypatch):
    before, view, added = _insert_case(host)
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    host.engine.ingest(view)
    _assert_cursor0(host.engine)
    assert host.rows() == before + [_key(row) for row in view if not row.get("timestamp") or row in added]
    assert _pairs(host.engine) == []
    assert not _counts(host.engine)["ledger.replay_pairs.cursor0"]


def test_09_relation_failure_keeps_rows_and_rolls_back_links(host, monkeypatch):
    before, view, added = _insert_case(host)
    store = host.engine._store
    original = store.add_message_relations
    attempted = []

    def fail(groups):
        if any(row[1] == "possible_replay_of" for group in groups for row in group):
            assert store._conn.in_transaction
            original(groups[:1])  # A partial successful write must also be rolled back.
            attempted.append(True)
            raise sqlite3.OperationalError("injected relation-write failure")
        return original(groups)

    monkeypatch.setattr(store, "add_message_relations", fail)
    host.engine.ingest(view)
    assert attempted == [True]
    assert host.rows() == before + [_key(row) for row in view if not row.get("timestamp") or row in added]
    assert _pairs(host.engine) == []
    assert _counts(host.engine)["ledger.write_failed"] == 1
    assert not _counts(host.engine)["ledger.replay_pairs.cursor0"]
