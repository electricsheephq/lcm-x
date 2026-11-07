"""I1/I2: a session-end todo snapshot must not become a silent replay copy."""

from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX
from tests.harness import engine_context, stored_rows


def _links(engine):
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone():
        return []
    return list(conn.execute(
        "SELECT store_id,kind,related_store_id FROM message_relations ORDER BY store_id,kind"))


@pytest.mark.parametrize("stamped", [True, False])
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_session_end_todo_replay(tmp_path, monkeypatch, stamped, nonzero, last, ledger):
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    todo = {"role": "user", "content": _PRESERVED_TODO_CONTEXT_PREFIX + "\n- [ ] 1. task (pending)",
            "timestamp": 103.0}
    view = [{"role": "user", "content": "U1", "timestamp": 101.0},
            {"role": "assistant", "content": "A1", "timestamp": 102.0},
            todo, {"role": "user", "content": "U2", "timestamp": 104.0}]
    if not stamped:
        todo.pop("timestamp")
    if last:
        view[-2:] = [view[-1], todo]
    new = {"role": "assistant", "content": "A2", "timestamp": 105.0}
    with engine_context(tmp_path, window=200_000, embeddings_enabled=False,
                        deferred_maintenance_enabled=False) as engine:
        if nonzero:
            engine.ingest(deepcopy(view[:2]))
        assert engine._ingest_cursor == (2 if nonzero else 0)
        engine.on_session_end("S", deepcopy(view))
        before = stored_rows(engine)
        assert sum(row["content"] == todo["content"] for row in before) == 1
        engine.on_session_start("S", context_length=200_000, platform="cli")
        engine.ingest(deepcopy(view + [new]))
        rows, links = stored_rows(engine), _links(engine)
        print("rows", [(r["store_id"], r["role"]) for r in rows])
        print("relations", links)
        assert engine._ingest_failure_count == 0
        assert rows[:len(before)] == before
        assert sum(row["content"] == new["content"] for row in rows) == 1
        copies = [r["store_id"] for r in rows
                  if r["content"] == todo["content"] and (not stamped or r["observed_at"] == todo["timestamp"])]
        replay = [(s, t) for s, kind, t in links if kind == "possible_replay_of" and s in copies]
        assert len(copies) == 1 or len(copies) == 2 and replay == [(copies[1], copies[0])]
        by_id = {row["store_id"]: row for row in rows}
        assert all(by_id[s]["content"] == by_id[t]["content"]
                   for s, kind, t in links if kind == "possible_replay_of")


@pytest.mark.parametrize("ledger", ["shadow", "off"])
@pytest.mark.parametrize("change", ["bytes", "stamp"])
def test_new_todo_snapshot_is_not_linked(tmp_path, monkeypatch, ledger, change):
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    todo = {"role": "user", "content": _PRESERVED_TODO_CONTEXT_PREFIX + "\n- [ ] 1. task (pending)",
            "timestamp": 103.0}
    fresh = dict(todo)
    if change == "bytes":
        fresh["content"] += "\n- [ ] 2. other task (pending)"
    else:
        fresh["timestamp"] = 106.0
    view = [{"role": "user", "content": "U1", "timestamp": 101.0},
            {"role": "assistant", "content": "A1", "timestamp": 102.0}, todo]
    with engine_context(tmp_path, window=200_000, embeddings_enabled=False,
                        deferred_maintenance_enabled=False) as engine:
        engine.on_session_end("S", deepcopy(view))
        before = stored_rows(engine)
        engine.on_session_start("S", context_length=200_000, platform="cli")
        engine.ingest(deepcopy(view + [{"role": "user", "content": "U2", "timestamp": 105.0}, fresh]))
        rows = stored_rows(engine)
        assert engine._ingest_failure_count == 0
        assert rows[:len(before)] == before
        added = [row for row in rows[len(before):]
                 if row["content"] == fresh["content"] and row["observed_at"] == fresh["timestamp"]]
        assert len(added) == 1
        assert not any(s == added[0]["store_id"] and kind == "possible_replay_of" for s, kind, _t in _links(engine))
