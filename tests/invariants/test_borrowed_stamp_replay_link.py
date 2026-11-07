"""One user send stored twice because the host showed it on a borrowed stamp.

After a compaction the host can show the next user prompt P on the stamp of the long prompt T that was in
flight (the borrowed stamp). LCM then stores P twice: once at its own stamp and once at T's stamp.
- 579 order: the own-stamp copy is stored at the turn start, the borrowed copy at the turn end.
- 383 order: the borrowed copy is stored at the turn end, the own-stamp copy after a restart.
The later copy carries one ``possible_replay_of`` link to the earlier one. Nothing is skipped or rewritten.
Controls: a genuine repeat, a host view that still shows the earlier copy, two candidates, stamps 1 µs
apart, and another session all stay unlinked.
"""

from collections import Counter
from copy import deepcopy

import pytest

from tests.invariants.test_issue_2378_restamp_link import all_rows, build as build, history, replay_links

pytestmark = pytest.mark.issue(2055)

T = "please draft the full migration plan for all four regional databases in detail"
P = "now summarise the rollback steps only"
NEW = "a distinct later turn"
T_LONG = 1794034596.8832  # the in-flight long prompt's stamp: the one P borrows
P_OWN = 1794034680.7996


def compacted(engine, before):
    view = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    return view


def restart(engine, build, cold, view):
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        return build()
    engine.on_session_start("S", conversation_id="conversation", platform="cli")
    return engine


def ids(engine, content):
    return [row["store_id"] for row in all_rows(engine) if row["content"] == content]


def check(engine, before_ids, *, copies, expected):
    """No stored row is lost, the links are exactly ``expected``, and no source has two links."""
    rows, links = all_rows(engine), replay_links(engine)
    by_id = {row["store_id"]: row for row in rows}
    evidence = {"copies": copies, "relations": sorted(links)}
    assert set(before_ids) <= set(by_id), evidence
    assert {(a, b) for a, b in links if a in copies or b in copies} == expected, evidence
    assert all(count == 1 for count in Counter(a for a, _b in links).values()), evidence
    assert all((by_id[a]["role"], by_id[a]["content"].strip()) == (by_id[b]["role"], by_id[b]["content"].strip())
               for a, b in links), evidence


def borrowed_first(build, cold, *, borrowed=T_LONG, keep_old=False, earlier=None):
    """383 order: P stored at the turn end on T's stamp, then at its own stamp after a restart."""
    engine = build()
    head = history(False) + ([{"role": "user", "content": P, "timestamp": earlier},
                              {"role": "assistant", "content": "done", "timestamp": earlier + 1}]
                             if earlier is not None else [])
    before = head + [{"role": "user", "content": T, "timestamp": T_LONG}]
    engine.ingest(deepcopy(before))
    view = compacted(engine, before)
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_LONG + 1},
             {"role": "user", "content": P, "timestamp": borrowed}]
    engine.ingest(deepcopy(view))
    stored = [row["store_id"] for row in all_rows(engine)]
    engine = restart(engine, build, cold, view)
    # The reload no longer holds the host's in-memory compress note, and shows P at its own stamp.
    view = [row for row in view[:-2] if row.get("content") != "compress finished"] + (view[-1:] if keep_old else [])
    view += [{"role": "user", "content": P, "timestamp": P_OWN},{"role": "assistant", "content": "rollback: restore, verify", "timestamp": P_OWN + 5},
             {"role": "user", "content": NEW, "timestamp": P_OWN + 10}]
    engine.ingest(deepcopy(view))
    assert len(ids(engine, NEW)) == 1
    assert len(ids(engine, T)) == 1
    return engine, stored


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_579_own_stamp_first_then_borrowed_copy_links(build, cold):
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_LONG}]
    engine.ingest(deepcopy(before))
    view = compacted(engine, before)
    # Turn start: P at its own stamp.
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_LONG + 1},
             {"role": "user", "content": P, "timestamp": P_OWN}]
    engine.ingest(deepcopy(view))
    stored = [row["store_id"] for row in all_rows(engine)]
    own = ids(engine, P)
    assert len(own) == 1
    if cold:
        engine.shutdown()
        engine = build()
    # Turn end: the host's list carries P on T's stamp and no longer holds T.
    host = [row for row in view if row.get("content") != T]
    next(row for row in host if row.get("content") == P)["timestamp"] = T_LONG
    host += [{"role": "assistant", "content": "rollback: restore, verify", "timestamp": P_OWN + 5},
             {"role": "user", "content": NEW, "timestamp": P_OWN + 10}]
    engine.ingest(deepcopy(host))
    copies = ids(engine, P)
    assert len(copies) == 2 and copies[0] == own[0], copies
    assert len(ids(engine, T)) == 1 and len(ids(engine, NEW)) == 1
    check(engine, stored, copies=copies, expected={(copies[1], copies[0])})


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_383_borrowed_copy_first_then_own_stamp_after_restart_links(build, cold):
    engine, stored = borrowed_first(build, cold)
    copies = ids(engine, P)
    assert len(copies) == 2, copies
    check(engine, stored, copies=copies, expected={(copies[1], copies[0])})


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_genuine_repeat_at_fresh_stamps_stays_unlinked(build, cold):
    engine = build()
    view = history(False) + [{"role": "user", "content": P, "timestamp": P_OWN - 50},
                             {"role": "assistant", "content": "rollback: restore, verify", "timestamp": P_OWN - 45}]
    engine.ingest(deepcopy(view))
    stored = [row["store_id"] for row in all_rows(engine)]
    engine = restart(engine, build, cold, view)
    view += [{"role": "user", "content": P, "timestamp": P_OWN},
             {"role": "assistant", "content": "same again", "timestamp": P_OWN + 5}]
    engine.ingest(deepcopy(view))
    copies = ids(engine, P)
    assert len(copies) == 2, copies
    check(engine, stored, copies=copies, expected=set())


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_earlier_copy_still_shown_at_its_stamp_vetoes(build, cold):
    """The host still shows P at T's stamp and adds P at a new stamp: a real repeat, never linked."""
    engine, stored = borrowed_first(build, cold, keep_old=True)
    copies = ids(engine, P)
    assert len(copies) == 2, copies
    check(engine, stored, copies=copies, expected=set())


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_two_candidate_earlier_copies_never_link(build, cold):
    engine, stored = borrowed_first(build, cold, earlier=P_OWN - 500)
    copies = ids(engine, P)
    assert len(copies) == 3, copies
    assert not any(a == copies[2] for a, _b in replay_links(engine)), sorted(replay_links(engine))
    check(engine, stored, copies=copies, expected={(a, b) for a, b in replay_links(engine) if a != copies[2]})


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_stamps_one_microsecond_apart_are_not_borrowed(build, cold):
    engine, stored = borrowed_first(build, cold, borrowed=T_LONG + 1e-6)
    copies = ids(engine, P)
    assert len(copies) == 2, copies
    stamps = {row["store_id"]: row["observed_at"] for row in all_rows(engine)}
    assert stamps[copies[0]] == T_LONG + 1e-6 != T_LONG, copies
    check(engine, stored, copies=copies, expected=set())


def test_a_row_in_another_session_never_links(build):
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_LONG}]
    engine.ingest(deepcopy(before))
    view = compacted(engine, before)
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_LONG + 1},
             {"role": "user", "content": P, "timestamp": T_LONG}]
    engine.ingest(deepcopy(view))
    stored = [row["store_id"] for row in all_rows(engine)]
    engine.on_session_end("S", deepcopy(view))
    engine.on_session_start("S2", conversation_id="conversation", platform="cli")
    engine.ingest([{"role": "user", "content": T, "timestamp": T_LONG + 900},
                   {"role": "assistant", "content": "ok", "timestamp": T_LONG + 901},
                   {"role": "user", "content": P, "timestamp": P_OWN}])
    copies = ids(engine, P)
    assert len(copies) == 2, copies
    sessions = {row["store_id"]: row["session_id"] for row in all_rows(engine)}
    assert [sessions[sid] for sid in copies] == ["S", "S2"]
    check(engine, stored, copies=copies, expected=set())
