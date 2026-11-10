"""#2377/#2443: borrowed user links and tool attribution share one ingest safely."""
from collections import Counter

import pytest

from tests.invariants._ingest_shapes_2377 import U, V, X, call, ingest_shape, relations
from tests.invariants.test_issue_2163_compacted_cursor import build as build

pytestmark = pytest.mark.issue(2377)


@pytest.mark.parametrize("borrowed_first", [False, True], ids=["own-first", "borrowed-first"])
@pytest.mark.parametrize("fault", [False, True], ids=["linked", "computation-fault"])
def test_borrowed_stamp_and_tool_replay_share_an_ingest(build, monkeypatch, borrowed_first, fault):
    holder = dict(role="user", content="the prompt whose stamp is borrowed", timestamp=1.0)
    prompt = dict(role="user", content="the next prompt", timestamp=1.0 if borrowed_first else 3.0)
    later = dict(prompt, timestamp=3.0 if borrowed_first else 1.0)
    configured = []

    def configure(engine):
        configured.append(engine)
        if fault and len(configured) == 2:
            def broken(*args, **kwargs):
                raise ValueError("synthetic link-computation fault")
            monkeypatch.setattr(engine, "_plan_tool_replay_links", broken)

    run = ingest_shape(build, monkeypatch,
                       [U, holder, call(2), dict(X, timestamp=2), prompt],
                       [U, call(4), dict(X, timestamp=4), later, V], configure=configure)
    run.check({4})
    assert set(run.inserted) == {1, 2, 3, 4}
    engine, source, target = run.engine, run.inserted[3], run.before[4]["store_id"]
    rels = relations(engine)
    assert all(n == 1 for n in Counter(s for s, kind, _ in rels if kind == "possible_replay_of").values())
    if fault:
        # The failed plan marks EVERY batch row, including the user copy. The borrowed
        # linker must leave that source pending, rather than add another attribution.
        assert {(s, kind, t) for s, kind, t in rels if s in run.inserted.values()} == {
            (s, "link_pending", None) for s in run.inserted.values()}
    else:
        assert {(s, t) for s, kind, t in rels if kind == "possible_replay_of"} == {
            (run.inserted[1], run.before[2]["store_id"]),
            (run.inserted[2], run.before[3]["store_id"]), (source, target)}
        conn = engine._store._conn
        assert conn.execute("SELECT observed_at FROM message_relations WHERE store_id=? "
                            "AND kind='possible_replay_of'", (source,)).fetchone() == (1.0,)
        assert not conn.execute("SELECT 1 FROM message_relations WHERE related_store_id=? "
                                "AND kind='possible_replay_of' AND observed_at IS NULL", (target,)).fetchone()
