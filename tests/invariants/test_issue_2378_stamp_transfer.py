"""A post-compress prompt inherits the prior turn's stamp, then returns at its own."""

from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX
from tests.invariants.test_issue_2378_restamp_link import (
    T, T_COMMIT, T_REQ, all_rows, build as build, history, replay_links,
)

pytestmark = pytest.mark.issue(2378)
P = "please check the next deployment step now"
NEW = "a distinct new turn after the rebind"


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize("restamp_t", [False, True], ids=["kept-T", "restamped-T"])
@pytest.mark.parametrize("retained_p", [False, True], ids=["transferred-P", "new-repeat-P"])
def test_next_prompt_stamp_transfer_is_not_silent(build, cold, restamp_t, retained_p):
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    assert sum(row["content"] == P for row in all_rows(engine)) == 1
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build()
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    if retained_p:
        view.append({"role": "user", "content": P, "timestamp": T_COMMIT})
    else:
        next(row for row in view if row.get("content") == P)["timestamp"] = T_COMMIT
    if restamp_t:
        next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    copies = [row["store_id"] for row in rows if row["content"] == P]
    # Content-free failure evidence: ids and relation kinds only.
    evidence = {"rows": copies,
                "relations": [(a, "possible_replay_of", b) for a, b in sorted(links)]}
    assert sum(row["content"] == NEW for row in rows) == 1, evidence
    assert len(copies) in (1, 2), evidence
    expected = set() if retained_p or len(copies) == 1 else {(copies[1], copies[0])}
    assert {(a, b) for a, b in links if a in copies} == expected, evidence
    by_id = {row["store_id"]: row for row in rows}
    assert all((by_id[a]["role"], by_id[a]["content"].strip()) ==
               (by_id[b]["role"], by_id[b]["content"].strip()) for a, b in links), evidence
    assert len({a for a, _b in links}) == len(links), evidence


@pytest.mark.xfail(strict=True, reason="625: TODO composite is scaffold before R2 "
                   "(engine.py:6663; identity_anchor.py:312), so no composite witness is recorded")
def test_625_todo_prompt_composite_with_stamp_transfer_copy_is_replay(build):
    engine = build()
    todo = _PRESERVED_TODO_CONTEXT_PREFIX + "\n- [>] 1. check deployment (in_progress)"
    head = history(False)
    parts = [{"role": "user", "content": todo, "timestamp": T_REQ},
             {"role": "user", "content": P, "timestamp": T_REQ + 1},
             {"role": "user", "content": P, "timestamp": T_REQ + 2}]
    engine.ingest(deepcopy(head + parts))
    before = all_rows(engine)
    assert sum(row["content"] == P for row in before) == 2
    assert sum(row["content"] == todo for row in before) == 1
    engine.on_session_end("S", deepcopy(head + parts))
    engine.on_session_start("S", conversation_id="conversation", platform="cli")
    composite = {"role": "user", "content": todo + "\n\n" + P, "timestamp": T_REQ}
    engine.ingest(deepcopy(head + [composite,
                                  {"role": "user", "content": NEW, "timestamp": T_COMMIT}]))
    rows = all_rows(engine)
    assert sum(row["content"] == NEW for row in rows) == 1
    assert not any(row["content"] == composite["content"] for row in rows)
    relations = list(engine._store._conn.execute(
        "SELECT store_id,kind,related_store_id FROM message_relations WHERE kind='composite'"))
    ids = {row["store_id"] for row in before if row["content"] in (todo, P)}
    assert len(relations) == 2
    assert all(a in ids and b in ids for a, _kind, b in relations)
