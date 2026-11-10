"""#2377 (#2414 threads eXR/lmG): a NON-transient relation-WRITE fault (ValueError from
add_message_relations) during the re-stamped replay ingest, then two unstamped occurrences.

The class invariant: a stored copy is never left unmarked. Every stored tool row whose skip-proof
identity repeats an earlier stored row carries possible_replay_of or link_pending, written in the
transaction that stored it; when neither can be written, that ingest fails and stores nothing."""
from copy import deepcopy
from collections import defaultdict

import pytest

from tests.harness import make_engine
from tests.invariants._ingest_shapes_2377 import U, X, call, relations

pytestmark = pytest.mark.issue(2377)


def _engine(folder, main_mode):
    engine = make_engine(folder, window=200_000, session_id="S", conversation_id="conversation",
                         fresh_tail_count=9, leaf_chunk_tokens=400,
                         large_output_externalization_enabled=False,
                         large_output_active_replay_stubbing_enabled=False,
                         temporal_rollups_enabled=False, embeddings_enabled=False,
                         deferred_maintenance_enabled=False)
    if main_mode and hasattr(engine, "_main_tool_segment_skip_ceiling"):
        def main_scan(messages, cursor, *, consumed_store_ids=(), consumed_pairs=None):
            engine._tool_cursor_spent, engine._tool_borrow_proofs = set(), {}
            if not engine._session_id or cursor >= len(messages):
                return set()
            tail = engine._store.get_session_tail(engine._session_id, limit=max(len(messages) * 4, 64))
            return set(engine._main_tool_segment_skip_ceiling(
                messages, cursor, consumed_store_ids=consumed_store_ids,
                consumed_pairs=consumed_pairs, tail_rows=tail))
        engine._replayed_tool_segment_indexes_after_cursor = main_scan
        engine._prepare_tool_replay_links = lambda *a, **kw: []
    return engine


def unmarked_copies(engine):
    """Stored tool rows whose proof identity repeats an earlier stored row, with no mark."""
    marked = {sid for sid, _kind, _target in relations(engine)}
    seen, unmarked = set(), []
    for row in engine._store.get_session_messages("S"):
        if row["role"] != "tool" and not row.get("tool_calls"):
            continue
        identity = (row.get("conversation_id"), engine._message_replay_identity(
            row, stored_row=True, allow_pruned_tool_alias=False))
        if identity in seen and row["store_id"] not in marked:
            unmarked.append(row["store_id"])
        seen.add(identity)
    return unmarked


def run(plan, folder, main_mode, fault_scope):
    """fault_scope: 'all' = every add_message_relations call in the faulted ingest raises ValueError;
    'link' = only the #2377 tool-link writer's possible_replay_of write raises."""
    folder.mkdir(parents=True, exist_ok=True)
    stored, log = defaultdict(list), []
    for number, step in enumerate(plan):
        engine = _engine(folder, main_mode)
        write = engine._store_ingest_rows

        def capture(write=write, **kwargs):
            ids = write(**kwargs)
            for (idx, _row), sid in zip(kwargs["messages_to_store_with_index"], ids):
                stored[(number, idx)].append(sid)
            return ids
        engine._store_ingest_rows = capture
        with pytest.MonkeyPatch.context() as patch:
            if step["cursor"]:
                at = step["cursor"]
                patch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, at=at, **kw: at)
                patch.setattr(engine, "_ledger_cold_cursor", lambda *a, at=at, **kw: dict(k=at, pairs={}))
            if step.get("fault"):
                real = engine._store.add_message_relations

                def broken(groups, *a, **kw):
                    raise ValueError("deterministic relation-write fault")

                def link_only(groups, *a, **kw):
                    if any(g and g[0][1] == "possible_replay_of" for g in groups):
                        raise ValueError("deterministic relation-write fault")
                    return real(groups, *a, **kw)
                if fault_scope == "all":
                    patch.setattr(engine._store, "add_message_relations", broken)
                elif not main_mode and hasattr(engine, "_write_tool_replay_links"):  # main has no tool writer
                    real_writer = engine._write_tool_replay_links

                    def tool_only(plan_, ids, after_commit):
                        with pytest.MonkeyPatch.context() as inner:
                            inner.setattr(engine._store, "add_message_relations", link_only)
                            return real_writer(plan_, ids, after_commit)
                    patch.setattr(engine, "_write_tool_replay_links", tool_only)
            engine.ingest(deepcopy(step["view"]))
        rows = engine._store.get_session_messages("S")
        log.append(dict(ingest=number, rows=[(r["store_id"], r["role"], r["content"], r["observed_at"]) for r in rows],
                        relations=relations(engine), failures=engine._ingest_failure_count,
                        unmarked=[] if main_mode else unmarked_copies(engine)))
        engine.shutdown()
    return dict(stored), log


A_X_2 = [U, call(2.0), dict(X, timestamp=2.0)]
NEXT = {"role": "user", "content": "next", "timestamp": 1000.0}


def plans():
    pair = [call(), dict(X)]
    connector = [dict(view=A_X_2, cursor=0),
                 dict(view=[U, call(4.0), dict(X, timestamp=4.0)], cursor=1, fault=True),
                 dict(view=[U, *pair, *pair], cursor=1)]
    neondiff = [dict(view=A_X_2, cursor=0),
                dict(view=[U, call(4.0), dict(X, timestamp=4.0), NEXT], cursor=1, fault=True),
                dict(view=[U, *pair, *pair, dict(NEXT, content="next2", timestamp=2000.0)], cursor=1)]
    # The new pair follows NEXT, so the unmarked @4 copy (stored before NEXT) cannot stand in for it.
    ordered = [dict(view=A_X_2, cursor=0),
               dict(view=[U, call(4.0), dict(X, timestamp=4.0), NEXT], cursor=1, fault=True),
               dict(view=[U, *pair, NEXT, *pair, dict(NEXT, content="next2", timestamp=2000.0)], cursor=1)]
    return dict(connector=connector, neondiff=neondiff, ordered=ordered)


def _new(shape):
    # The genuinely new identical occurrence in the third view.
    return {(2, 4), (2, 5)} if shape == "ordered" else {(2, 3), (2, 4)}


@pytest.mark.parametrize("shape", ["connector", "neondiff", "ordered"])
def test_relation_write_fault_copy_is_never_skip_capacity(tmp_path, shape):
    plan = plans()[shape]
    v7, v7_log = run(plan, tmp_path / "v7", False, "link")
    main, main_log = run(plan, tmp_path / "main", True, "link")
    assert all(entry["failures"] == 0 for entry in v7_log + main_log), (v7_log, main_log)
    lost = sorted(set(main) - set(v7))
    assert _new(shape) <= set(main), dict(main=sorted(main), main_log=main_log)
    assert not lost, dict(lost=lost, v7=sorted(v7), main=sorted(main), v7_log=v7_log[-1], main_log=main_log[-1])
    # The faulted copy is marked link_pending in the same transaction, never left unmarked.
    assert {kind for _sid, kind, _target in v7_log[1]["relations"]} == {"link_pending"}, v7_log[1]
    assert all(not entry["unmarked"] for entry in v7_log), v7_log


@pytest.mark.parametrize("shape", ["connector", "neondiff", "ordered"])
def test_double_fault_never_commits_an_unmarked_copy(tmp_path, shape):
    """Neither the link nor link_pending can be written: that ingest fails and stores nothing;
    the next ingest of the same view stores the copy with its mark."""
    plan = plans()[shape]
    _stored, log = run(plan, tmp_path / "v7", False, "all")
    assert log[1]["failures"] == 1, log[1]
    # Rolled back as a unit: the store is exactly what the first ingest left.
    assert log[1]["rows"] == log[0]["rows"] and log[1]["relations"] == [], log[1]
    # The retry: the view the failed ingest held arrives again with no fault.
    retry = [plan[0], plan[1], dict(plan[1], fault=False), plan[2]]
    stored, log = run(retry, tmp_path / "retry", False, "all")
    assert [entry["failures"] for entry in log] == [0, 1, 0, 0], log
    assert any(number == 2 for number, _idx in stored), sorted(stored)
    assert log[2]["relations"], log[2]
    assert all(not entry["unmarked"] for entry in log), log
    assert {(3, idx) for _n, idx in _new(shape)} <= set(stored), dict(stored=sorted(stored), log=log[-1])


@pytest.mark.parametrize("shape", ["connector", "neondiff", "ordered"])
def test_native_tree_stores_the_new_pair(tmp_path, shape):
    stored, log = run(plans()[shape], tmp_path / "native", False, "link")
    assert _new(shape) <= set(stored), dict(stored=sorted(stored), log=log[-1])
