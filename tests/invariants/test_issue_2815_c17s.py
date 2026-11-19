"""The UID-located durable tail bounds cold replay even without a cut budget."""

import logging
from copy import deepcopy

import pytest

from tests.invariants.test_issue_2815_c9 import _contents, _mode, _open
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _rows

_BUDGET_DISABLED = "LCM UID cut budget disabled"


def _row(role, content, uid, timestamp):
    return {"role": role, "content": content, "message_uid": uid, "timestamp": timestamp}


def _turn(prefix, timestamp):
    return [_row("user", "Q", f"{prefix}-q", timestamp), _row("assistant", "OK", f"{prefix}-a", timestamp + 1)]


def _seed(path, stored, *, unbound=()):
    _state_db(path, [("S", None, None)])
    with _open(path) as seed:
        ids = [seed._store.append("S", row, conversation_id="conv") for row in stored]
        lineage = seed._host_uid_lineage_key()[0]
        seed._store.add_host_uid_bindings(lineage, [
            (store_id, row["message_uid"], "canonical", "stored_new")
            for store_id, row in zip(ids, stored)
            if row["message_uid"] not in unbound
        ])
    return ids


def _repeated_view(path, fresh):
    stored = _turn("s1", 1000.0) + _turn("s2", 1002.0)
    added = _turn("f", 2000.0) + [
        _row("user" if i % 2 == 0 else "assistant", f"Distinct fresh payload {i}", f"f-{i}", 2002.0 + i)
        for i in range(fresh - 2)
    ]
    return stored, added, _seed(path, stored)


def _cold_ingest(engine, monkeypatch, view):
    cursors = []
    reconcile = engine._reconcile_ingest_cursor_from_store

    def record(*args, **kwargs):
        cursor = reconcile(*args, **kwargs)
        cursors.append(cursor)
        return cursor

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_reconcile_ingest_cursor_from_store", record)
        engine.ingest(deepcopy(view))
    return cursors


def _bindings(engine, messages):
    lineage = engine._host_uid_lineage_key()[0]
    return engine._store.host_uid_bindings_for(lineage, [row["message_uid"] for row in messages])


def _assert_rows_and_bindings(rows, bindings, stored, added, seed_ids):
    assert _contents(rows) == _contents(stored + added)
    ids = [row["store_id"] for row in rows]
    assert ids[:len(seed_ids)] == seed_ids
    assert len(set(ids)) == len(ids)
    assert bindings == {
        row["message_uid"]: [(store_id, "canonical")]
        for row, store_id in zip(added, ids[len(seed_ids):])
    }


# Round c17t (parity): this cut path is e737bae9's again, so the known AP1 loss is back at N >= 65. It is a named
# residual under lcm-x #2962, which the PR 2963 port (uid-only rejection of a contradicted full-replay cut) fixes.
_AP1 = pytest.mark.xfail(strict=True, raises=AssertionError,
                         reason="AP1 (lcm-x #2962): fresh rows repeating the stored tail's last turn are lost at N >= 65")

@pytest.mark.parametrize("anchor", ["on", "off"])
@pytest.mark.parametrize("fresh", [62, 63, 64, pytest.param(65, marks=_AP1), pytest.param(66, marks=_AP1)])
def test_cold_ingest_keeps_repeated_fresh_turn(tmp_path, monkeypatch, fresh, anchor):
    _mode(monkeypatch, True)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    stored, added, seed_ids = _repeated_view(tmp_path, fresh)
    with _open(tmp_path) as cold:
        accepted = _cold_ingest(cold, monkeypatch, [*stored[2:], *added])
        rows = _rows(cold)
        bindings = _bindings(cold, added)

    assert len(rows) == 4 + fresh
    _assert_rows_and_bindings(rows, bindings, stored, added, seed_ids)
    assert accepted == [2]


@pytest.mark.parametrize("fresh", [64, pytest.param(65, marks=_AP1)])
def test_repeated_fresh_turn_stays_stored_through_warm_ingest_and_reopen(tmp_path, monkeypatch, fresh):
    _mode(monkeypatch, True)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    stored, added, seed_ids = _repeated_view(tmp_path, fresh)
    view = [*stored[2:], *added]
    next_row = _row("user" if len(view) % 2 == 0 else "assistant", "One more row", "f-next", 3000.0)
    extended = [*view, next_row]
    with _open(tmp_path) as cold:
        accepted = _cold_ingest(cold, monkeypatch, view)
        cold.ingest(deepcopy(extended))
        warm_rows = _rows(cold)
        warm_cursor = cold._ingest_cursor
    with _open(tmp_path) as reopened:
        reopened.ingest(deepcopy(extended))
        rows = _rows(reopened)
        bindings = _bindings(reopened, [*added, next_row])
        reopened_cursor = reopened._ingest_cursor

    assert len(rows) == 4 + fresh + 1
    _assert_rows_and_bindings(rows, bindings, stored, [*added, next_row], seed_ids)
    assert rows == warm_rows
    assert accepted == [2]
    assert warm_cursor == reopened_cursor == len(extended)


@pytest.mark.parametrize("legacy, unbound_count", [(10, 0), (40, 0), (59, 1), (70, 12)])
def test_cold_ingest_with_unbound_legacy_stretch(tmp_path, monkeypatch, caplog, legacy, unbound_count):
    _mode(monkeypatch, True)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    caplog.set_level(logging.INFO, logger="hermes_lcm.reconcile")
    head = [_row("user", "Head question", "s-h0", 900.0), _row("assistant", "Head answer", "s-h1", 901.0)]
    old = [
        _row("user" if i % 2 == 0 else "assistant", f"Legacy row {i}", f"l-{i}", 910.0 + i)
        for i in range(legacy)
    ]
    stored = [*head, *old, *_turn("s1", 1000.0), *_turn("s2", 1002.0)]
    seed_ids = _seed(tmp_path, stored, unbound={row["message_uid"] for row in old})
    added = _turn("f", 2000.0)
    with _open(tmp_path) as cold:
        accepted = _cold_ingest(cold, monkeypatch, [*stored, *added])
        _assert_rows_and_bindings(_rows(cold), _bindings(cold, added), stored, added, seed_ids)

    assert accepted == [len(stored)]
    assert [line for line in caplog.messages if _BUDGET_DISABLED in line] == (
        [f"{_BUDGET_DISABLED}: unbound_host_rows={unbound_count}"] if unbound_count else []
    )


def test_cold_ingest_with_one_unbound_row_in_periodic_store(tmp_path, monkeypatch, caplog):
    _mode(monkeypatch, True)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    caplog.set_level(logging.INFO, logger="hermes_lcm.reconcile")
    stored = [row for i in range(40) for row in _turn(f"s-{i}", 1000.0 + 2 * i)]
    seed_ids = _seed(tmp_path, stored, unbound={stored[17]["message_uid"]})
    added = _turn("f", 2000.0)
    with _open(tmp_path) as cold:
        accepted = _cold_ingest(cold, monkeypatch, [*stored[-66:], *added])
        _assert_rows_and_bindings(_rows(cold), _bindings(cold, added), stored, added, seed_ids)

    assert accepted == [3]
    assert [line for line in caplog.messages if _BUDGET_DISABLED in line] == [
        f"{_BUDGET_DISABLED}: unbound_host_rows=1"
    ]
