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


def borrowed_first(build, cold, *, borrowed=T_LONG, keep_old=False, earlier=None, holder_shown=True):
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
    view = [row for row in view[:-2] if row.get("content") != "compress finished"
            and (holder_shown or row.get("content") != T)] + (view[-1:] if keep_old else [])
    view += [{"role": "user", "content": P, "timestamp": P_OWN},
             {"role": "assistant", "content": "rollback: restore, verify", "timestamp": P_OWN + 5},
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


@pytest.mark.parametrize("holder_shown", [True, False], ids=["holder-shown", "holder-gone"])
@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_383_borrowed_copy_first_then_own_stamp_after_restart_links(build, cold, holder_shown):
    """The stamp gate reads the store, so the link does not need the reload view to still show T."""
    engine, stored = borrowed_first(build, cold, holder_shown=holder_shown)
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


# The link never consumes E. A later copy R of E must still take its own link from the re-stamp pairing, the
# cursor-0 scan and the durable-recovery link, which each allow one consuming link per target.
def borrowed_links(engine):
    return {(a, b, stamp) for a, b, stamp in engine._store._conn.execute(
        "SELECT store_id, related_store_id, observed_at FROM message_relations WHERE kind='possible_replay_of'")}


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_the_link_records_its_borrowed_stamp(build, cold):
    engine, _stored = borrowed_first(build, cold)
    copies = ids(engine, P)
    assert borrowed_links(engine) == {(copies[1], copies[0], T_LONG)}


@pytest.mark.parametrize("mode,rebind", [("inplace", False), ("inplace", True), ("rotating", False)],
                         ids=["inplace-warm", "inplace-rebind", "rotating"])
def test_restamped_copy_still_links_to_a_target_of_a_borrowed_link(build, caplog, mode, rebind):
    """In place, the durable-recovery link pairs the re-stamped copy; on rotation, the re-stamp pairing does."""
    from bench.instruments.reliability.scorers.multiset import split_surplus
    from tests.invariants.test_issue_2378_restamp_link import T as T_RESTAMP, T_COMMIT, T_REQ, run

    made = {}

    def borrow(engine, _view):
        # E (T at its own stamp) already carries a borrowed-stamp link, in the linker's format.
        rows = all_rows(engine)
        made["target"] = next(row["store_id"] for row in rows if row["content"] == T_RESTAMP and row["observed_at"] == T_REQ)
        made["source"] = next(row["store_id"] for row in rows if row["role"] == "user" and row["observed_at"] == 1130.0)
        engine._store.add_message_relations([[(made["source"], "possible_replay_of", made["target"], None, 1130.0)]])
        if rebind:
            engine.on_session_start("S", conversation_id=engine._conversation_id, platform="cli")

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=borrow)
    restamped = [row["store_id"] for row in rows if row["content"] == T_RESTAMP and row["observed_at"] == T_COMMIT]
    assert len(restamped) == 1
    links = replay_links(engine)
    assert (restamped[0], made["target"]) in links, sorted(links)
    assert all(count == 1 for count in Counter(a for a, _b in links).values()), sorted(links)
    held = [row["store_id"] for row in rows if row["content"] == T_RESTAMP]
    evidence = {row["store_id"]: (row["session_id"], row["role"], row["content"]) for row in rows}
    split = split_surplus([(("user", T_RESTAMP), len(held) - 1, held, [], evidence)],
                          [(a, "possible_replay_of", b) for a, b in links])
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (len(held) - 1, 0), split


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_cursor0_copies_still_link_to_targets_of_borrowed_links(cold, tmp_path, monkeypatch):
    from hermes_lcm.engine import LCMEngine
    from tests.test_issue_1286_cleanup_rewrite_proof import SID, Host
    from tests.test_issue_2305_cursor0_replay_links import _assert_cursor0, _counts, _insert_case, _pairs

    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        monkeypatch.setattr(LCMEngine, name, lambda *args: None)
    host = Host(tmp_path, embeddings_enabled=False)
    try:
        _before, view, _added = _insert_case(host, cold=cold)
        rows = [row["store_id"] for row in host.engine._store.get_session_messages(SID)]
        # A borrowed-stamp link into every earlier row, in the linker's format (the borrowed stamp in observed_at).
        host.engine._store.add_message_relations([[(rows[0], "possible_replay_of", target, None, T_LONG)]
                                                  for target in rows[1:]])
        host.engine.ingest(view)
        _assert_cursor0(host.engine)
        assert _counts(host.engine)["ledger.replay_pairs.cursor0"] == 7
        assert len([pair for pair in _pairs(host.engine) if pair[0] != rows[0]]) == 7
    finally:
        host.close()


def test_durable_recovery_still_links_to_a_target_of_a_borrowed_link(tmp_path, monkeypatch):
    from tests.harness import ScriptedSummarizer, engine_context, pin_char_counter
    from tests.invariants.test_issue_1505_recovery_stamp import lineage, message, users

    pin_char_counter(monkeypatch)
    ScriptedSummarizer("outage").install(monkeypatch)
    lineage(tmp_path)
    current = message("[T32] rebuild the index " * 500, 20.)
    source = [message("[T31] earlier request", 10.), message("earlier reply", 11., "assistant"), current,
              {"role": "tool", "content": "orphan result", "tool_call_id": "orphan", "timestamp": 21.}]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(source)
        recovered = next(r["store_id"] for r in users(engine) if r["content"] == current["content"])
        first = next(r["store_id"] for r in users(engine) if r["content"] == source[0]["content"])
        engine._config.fresh_tail_count, engine._config.max_assembly_tokens = 64, 120
        view = list(engine.compress(source, current_tokens=100, force=True))
        assert engine._last_compression_status == "overflow_recovery"
        view[-1]["timestamp"] = 30.
        # A borrowed-stamp link into the recovered row, in the linker's format.
        engine._store.add_message_relations([[(first, "possible_replay_of", recovered, None, 10.)]])
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        view[-1]["content"] = current["content"]
        view.append(message("reply", 31., "assistant"))
        engine.ingest(deepcopy(view))
        copies = {r["store_id"] for r in users(engine) if r["content"] == current["content"]}
        linked = {r["store_id"] for r in engine._store.get_message_relations(list(copies), "possible_replay_of")
                  if r["related_store_id"] == recovered}
        assert len(copies) == 2 and copies - {recovered} == linked, (sorted(copies), sorted(linked))


def test_two_links_into_one_target_both_count_visible():
    """The kit's split_surplus needs no distinct targets: E kept, N and R both linked to E, both visible."""
    from bench.instruments.reliability.scorers.multiset import split_surplus

    evidence = {sid: ("S", "user", P) for sid in (1, 2, 3)}
    split = split_surplus([(("user", P), 2, [1, 2, 3], [], evidence)],
                          [(2, "possible_replay_of", 1), (3, "possible_replay_of", 1)])
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (2, 0), split


@pytest.mark.parametrize("n", [500, 2000])
def test_linker_reads_grow_linearly_on_a_bulk_ingest(build, monkeypatch, n):
    """A cold first ingest of n distinct user turns: the linker issues a constant number of reads per row."""
    import hermes_lcm.engine as engine_module

    engine, statements = build(), []
    link = engine_module.LCMEngine._link_borrowed_stamp_occurrences

    def counted(self, *args):
        self._store._conn.set_trace_callback(statements.append)
        try:
            return link(self, *args)
        finally:
            self._store._conn.set_trace_callback(None)
    monkeypatch.setattr(engine_module.LCMEngine, "_link_borrowed_stamp_occurrences", counted)
    turns = []
    for i in range(n):
        turns += [{"role": "user", "content": f"distinct turn {i}", "timestamp": 1000.0 + 2 * i},
                  {"role": "assistant", "content": f"reply {i}", "timestamp": 1001.0 + 2 * i}]
    engine.ingest(turns)
    assert len(ids(engine, f"distinct turn {n - 1}")) == 1
    assert len(statements) <= 3 * n, len(statements)


def test_known_limit_a_genuine_repeat_sharing_a_stamp_with_other_text_is_linked(build):
    """Characterization: a host that stamps two different user texts identically for a reason other than the
    borrow shows the borrowed-stamp signature, so a genuine repeat P is attributed to its earlier copy. The link
    only attributes: every row stays stored."""
    engine = build()
    first = history(False) + [{"role": "user", "content": P, "timestamp": 1200.0},
                              {"role": "assistant", "content": "done", "timestamp": 1201.0}]
    engine.ingest(deepcopy(first))
    stored = [row["store_id"] for row in all_rows(engine)]
    view = compacted(engine, first)
    view = [row for row in view if row.get("content") != P]  # P compacted out of the host view
    view += [{"role": "user", "content": T, "timestamp": 5000.0},
             {"role": "user", "content": P, "timestamp": 5000.0},
             {"role": "assistant", "content": "again", "timestamp": 5001.0}]
    engine.ingest(deepcopy(view))
    copies = ids(engine, P)
    assert len(copies) == 2, copies
    check(engine, stored, copies=copies, expected={(copies[1], copies[0])})


def test_one_way_a_consumed_target_takes_no_borrowed_link(build):
    """By design the borrowed linker treats any incoming link on E as taken, so N stays unlinked when another
    copy already holds E; the reverse never happens because a borrowed link does not consume E."""
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_LONG}]
    engine.ingest(deepcopy(before))
    view = compacted(engine, before)
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_LONG + 1},
             {"role": "user", "content": P, "timestamp": P_OWN}]
    engine.ingest(deepcopy(view))
    own = ids(engine, P)[0]
    other = next(row["store_id"] for row in all_rows(engine) if row["role"] == "user" and row["observed_at"] == 1130.0)
    engine._store.add_message_relations([[(other, "possible_replay_of", own, None, None)]])  # a consuming link
    host = [row for row in view if row.get("content") != T]
    next(row for row in host if row.get("content") == P)["timestamp"] = T_LONG
    host.append({"role": "assistant", "content": "rollback: restore, verify", "timestamp": P_OWN + 5})
    engine.ingest(deepcopy(host))
    copies = ids(engine, P)
    assert len(copies) == 2, copies
    assert not any(a == copies[1] for a, _b in replay_links(engine)), sorted(replay_links(engine))


def test_durable_recovery_never_gives_a_borrowed_source_a_second_link(tmp_path, monkeypatch):
    """The durable-recovery link runs after the borrowed linker; a source that already holds a link is skipped.
    The borrowed linker's write is forced here (the harness cannot borrow the recovered prompt's stamp)."""
    import hermes_lcm.engine as engine_module
    from tests.harness import ScriptedSummarizer, engine_context, pin_char_counter
    from tests.invariants.test_issue_1505_recovery_stamp import lineage, message, users

    pin_char_counter(monkeypatch)
    ScriptedSummarizer("outage").install(monkeypatch)
    lineage(tmp_path)
    current = message("[T32] rebuild the index " * 500, 20.)
    source = [message("[T31] earlier request", 10.), message("earlier reply", 11., "assistant"), current,
              {"role": "tool", "content": "orphan result", "tool_call_id": "orphan", "timestamp": 21.}]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(source)
        recovered = next(r["store_id"] for r in users(engine) if r["content"] == current["content"])
        engine._config.fresh_tail_count, engine._config.max_assembly_tokens = 64, 120
        view = list(engine.compress(source, current_tokens=100, force=True))
        assert engine._last_compression_status == "overflow_recovery"
        view[-1]["timestamp"] = 30.

    def borrowed_first(self, messages, stored_at):
        for idx, sid in stored_at.items():
            if messages[idx].get("content") == current["content"]:
                self._store.add_message_relations([[(sid, "possible_replay_of", recovered, None, 30.)]])
    monkeypatch.setattr(engine_module.LCMEngine, "_link_borrowed_stamp_occurrences", borrowed_first)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        view[-1]["content"] = current["content"]
        view.append(message("reply", 31., "assistant"))
        engine.ingest(deepcopy(view))
        copies = [r["store_id"] for r in users(engine) if r["content"] == current["content"]]
        links = list(engine._store._conn.execute(
            "SELECT store_id, related_store_id, observed_at FROM message_relations WHERE kind='possible_replay_of'"))
        assert len(copies) == 2 and links == [(copies[1], recovered, 30.)], (copies, links)
