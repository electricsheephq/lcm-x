"""#2305: annotate cursor-0 copies without changing the stored message stream."""

from collections import Counter, defaultdict
from copy import deepcopy
import sqlite3
import logging
import re

import pytest

from bench.instruments.reliability.scorers.multiset import split_surplus
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _strip_reasoning_blocks
from hermes_lcm.ingest_ledger import stripped_replay_content
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
        left, right = by_id[source], by_id[target]
        assert (left["role"], left["tool_call_id"], left["tool_calls"]) == (
            right["role"], right["tool_call_id"], right["tool_calls"])
        assert stripped_replay_content(left["content"]) == stripped_replay_content(right["content"])
    evidence = {sid: (row["session_id"], row["role"], row["content"]) for sid, row in by_id.items()}
    groups = [(key, max(0, len(held) - wanted[key]), held, [], evidence) for key, held in ids.items()]
    split = split_surplus(groups, [(s, "possible_replay_of", t) for s, t in pairs])
    assert split["silent_surplus_rows"] == 0
    assert split["visible_surplus_rows"] == surplus
    assert _counts(host.engine)["ledger.replay_pairs.cursor0"] == surplus
    assert host.engine.get_status()["ingest_ledger"]["ledger.replay_pairs.cursor0"] == surplus
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_stripped"] == 1


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
    previous_seq = store._conn.execute("SELECT max(view_seq) FROM ingest_views").fetchone()[0]
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
    new_ids = {row["store_id"] for row in store.get_session_messages(SID)[len(before):]}
    bound_ids = {row[0] for row in store._conn.execute("SELECT store_id FROM occurrence_bindings")}
    assert new_ids <= bound_ids
    assert store._conn.execute("SELECT max(view_seq) FROM ingest_views").fetchone()[0] > previous_seq
    assert _counts(host.engine)["ledger.bindings_written"] >= len(new_ids)
    assert _counts(host.engine)["ledger.views_written"] > 0


def test_10_different_reasoning_is_linked_and_stored(host):
    host.turn(reply="<think>old reasoning</think>answer")
    target = host.engine._store.get_session_messages(SID)[-1]["store_id"]
    sources = _ingest_copies(host, [{"role": "assistant", "content": "<think>new reasoning</think>answer"}])
    assert _pairs(host.engine) == [(sources[0], target)]
    assert host.rows()[-1][1] == "<think>new reasoning</think>answer"
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_stripped"] == 1


def test_11_exact_pass_spends_targets_before_stripped_pass(host):
    for reply in ("<think>oldest</think>ok", "ok", "<think>third</think>ok"):
        host.turn(reply=reply)
    targets = [r["store_id"] for r in host.engine._store.get_session_messages(SID) if r["role"] == "assistant"]
    copies = [{"role": "assistant", "content": text} for text in ("<think>new</think>ok", "ok", "ok")]
    sources = _ingest_copies(host, copies)
    assert _pairs(host.engine) == [(sources[0], targets[0]), (sources[1], targets[1]), (sources[2], targets[2])]
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_stripped"] == 2


@pytest.mark.parametrize("raw,expected", [
    ("<think>unclosed answer", "<think>unclosed answer"),
    ("<think>outer<think>inner</think>rest</think>answer", "rest</think>answer"),
    ("<think>a</think>x<thinking>b</thinking>y", "xy"),
    ("<think>a</think> answer ", " answer "),
])
def test_12_product_stripping_is_the_only_transform(host, raw, expected):
    assert _strip_reasoning_blocks(raw) == expected
    assert stripped_replay_content(raw) == expected
    host.turn(reply=raw)
    target = host.engine._store.get_session_messages(SID)[-1]["store_id"]
    sources = _ingest_copies(host, [{"role": "assistant", "content": expected}])
    assert _pairs(host.engine) == [(sources[0], target)]


def test_13_other_session_is_refused(host):
    other = host.bind("other-session")
    other.ingest([{"role": "assistant", "content": "<think>old</think>answer"}])
    host.turn(reply="unrelated")
    _ingest_copies(host, [{"role": "assistant", "content": "answer"}])
    assert _pairs(host.engine) == []


def test_14_stripped_stamp_veto(host, monkeypatch):
    stamp = host.stamp()
    host.live += [{"role": "assistant", "content": "<think>old</think>answer", "timestamp": stamp}]
    host.engine.ingest(host.live)
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    _ingest_copies(host, [{"role": "assistant", "content": "answer", "timestamp": stamp + 10}])
    assert _pairs(host.engine) == []
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 1


@pytest.mark.parametrize("text", ["<think>old</think>", "<think>old</think> \n"])
def test_15_all_reasoning_links_only_byte_exact(host, text):
    """#588 (release-manager ruling): a reasoning-only copy is evidence on a byte-exact match (exact pass only)."""
    host.turn(reply=text)
    sources = _ingest_copies(host, [{"role": "assistant", "content": text}])
    target = next(row["store_id"] for row in host.engine._store.get_session_messages(SID)
                  if row["content"] == text and row["store_id"] not in sources)
    assert _pairs(host.engine) == [(sources[0], target)]
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_stripped"] == 0
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 0


def test_15b_all_reasoning_with_other_bytes_never_links(host):
    """The stripped pass is unchanged: two reasoning-only rows whose bytes differ are not each other's evidence."""
    host.turn(reply="<think>old</think>")
    _ingest_copies(host, [{"role": "assistant", "content": "<think>new</think>"}])
    assert _pairs(host.engine) == []


@pytest.mark.parametrize("field,value", [("tool_call_id", "other"), ("tool_calls", [
    {"id": "other", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}])])
def test_16_tool_fields_must_match(host, field, value):
    host.turn(reply="<think>old</think>answer")
    _ingest_copies(host, [{"role": "assistant", "content": "answer", field: value}])
    assert _pairs(host.engine) == []


def _route_records(caplog):
    return [record for record in caplog.records if record.getMessage().startswith("LCM ingest route:")]


def test_17_recorder_cursor0_and_append(host, caplog):
    before, view, _added = _insert_case(host)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        host.engine.ingest(view)
    records = _route_records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.INFO
    match = re.fullmatch(r"LCM ingest route: session=(\S+) route=(.*?) cursor=(\d+) reason=(.*?) "
                         r"stored=(\d+)-(\d+) rows=(\d+) replay_pairs=(\d+) stripped=(\d+)", records[0].getMessage())
    assert match is not None
    new = host.engine._store.get_session_messages(SID)[len(before):]
    assert match.groups() == (SID, "persisted_batch", "0", "persisted ambiguous delta", str(new[0]["store_id"]),
                              str(new[-1]["store_id"]), str(len(new)), "7", "1")
    assert all(row["content"] not in records[0].getMessage() for row in new)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        host.engine.ingest([*view, {"role": "user", "content": "ordinary append", "timestamp": host.stamp()}])
    records = _route_records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.DEBUG
    assert "route=foreground" in records[0].getMessage()
    assert "rows=1 replay_pairs=0 stripped=0" in records[0].getMessage()


def test_18_cursor0_tool_turn_links_call_and_result(host):
    host.tool_turn()
    originals = host.engine._store.get_session_messages(SID)[-3:-1]
    copies = deepcopy(host.live[-3:-1])
    assert copies[0]["content"] == "" and copies[0]["tool_calls"]
    sources = _ingest_copies(host, copies)
    assert _pairs(host.engine) == list(zip(sources, [row["store_id"] for row in originals]))
    assert _counts(host.engine)["ledger.replay_pairs.cursor0"] == 2
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 0


@pytest.mark.parametrize("content", ["", "same output"])
def test_19_tool_names_without_call_ids_must_match(host, content):
    host.live += [{"role": "tool", "tool_name": name, "content": content} for name in ("search", "lookup")]
    host.engine.ingest(host.live)
    originals = host.engine._store.get_session_messages(SID)[-2:]
    assert [row["tool_name"] for row in originals] == ["search", "lookup"]
    sources = _ingest_copies(host, [{"role": "tool", "tool_name": name, "content": content}
                                    for name in ("lookup", "different")])
    # Empty unkeyed content stays unlinked, but a matching-name candidate is counted.
    assert _pairs(host.engine) == ([(sources[0], originals[1]["store_id"])] if content else [])
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == int(not content)


@pytest.mark.parametrize("field,value", [("tool_call_id", "call-empty"), ("tool_calls", [
    {"id": "call-empty", "type": "function", "function": {"name": "search", "arguments": "{}"}}])])
def test_20_empty_tool_key_is_one_to_one_and_counted(host, field, value):
    row = {"role": "assistant", "content": "", field: value}
    host.live.append(deepcopy(row))
    host.engine.ingest(host.live)
    target = host.engine._store.get_session_messages(SID)[-1]["store_id"]
    sources = _ingest_copies(host, [deepcopy(row), deepcopy(row)])
    assert _pairs(host.engine) == [(sources[0], target)]
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 1


def test_21_recorder_direct_off_current_session_end(tmp_path, caplog):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "s3-direct.db")))
    try:
        engine.on_session_start("ended", platform="cli", context_length=1_000)
        opener = {"role": "user", "content": "opener"}
        engine.ingest([opener])
        engine.on_session_start("foreground", platform="cli", context_length=1_000)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
            engine.on_session_end("ended", [opener, {"role": "assistant", "content": "final"}])
        rows = engine._store.get_range("ended")
        assert [row["content"] for row in rows] == ["opener", "final"]
        records = _route_records(caplog)
        assert len(records) == 1 and records[0].levelno == logging.INFO
        assert records[0].getMessage() == (
            f"LCM ingest route: session=ended route=S3 cursor=1 reason=off-current session end "
            f"stored={rows[1]['store_id']}-{rows[1]['store_id']} rows=1 replay_pairs=0 stripped=0")
    finally:
        engine.shutdown()


def test_22_earlier_rows_are_read_in_one_linear_scan(host):
    # The link pass reads the session's earlier rows once; a per-key join re-scans the key set for every row and
    # holds the ingest write lock for seconds on a tool-heavy session (#2335 review).
    host.tool_turn()
    copies = deepcopy(host.live[-3:-1])
    host.restart()
    statements = []
    host.engine._store._conn.set_trace_callback(statements.append)
    try:
        host.engine.ingest([{"role": "user", "content": "different head", "timestamp": host.stamp()}, *copies])
    finally:
        host.engine._store._conn.set_trace_callback(None)
    _assert_cursor0(host.engine)
    assert len(_pairs(host.engine)) == 2
    earlier = [sql for sql in statements if "FROM messages" in sql and "store_id<" in sql.replace(" ", "")]
    # #2055: the borrowed-stamp linker's reads (user rows only) are its own, and never a scan of the session's
    # rows: it reads earlier copies by identity digest and stamps by index.
    borrowed = [sql for sql in earlier if "role='user'" in sql]
    earlier = [sql for sql in earlier if sql not in borrowed]
    assert all("observed_at" in sql.split("WHERE", 1)[1] for sql in borrowed), borrowed
    assert len(earlier) == 1 and not any("json_extract(k.value" in sql for sql in statements)


def test_23_many_identical_copies_link_one_to_one_oldest_first(host):
    # #2335 review: N copies of one payload link one-to-one, oldest target first. This checks the pairing only;
    # the matcher's linear cost is measured separately.
    host.turn(reply="ok")
    host.live += [{"role": "user", "content": "same payload"} for _ in range(40)]
    host.engine.ingest(host.live)
    targets = [r["store_id"] for r in host.engine._store.get_session_messages(SID) if r["content"] == "same payload"]
    assert len(targets) == 40
    sources = _ingest_copies(host, [{"role": "user", "content": "same payload"} for _ in range(40)])
    assert len(sources) == 40
    assert _pairs(host.engine) == list(zip(sources, targets))  # every copy linked, oldest target first


def test_24_stamped_copies_take_the_oldest_allowed_target(host, monkeypatch):
    # #2335 review: candidates are indexed by stamp, so vetoed rows are skipped without a rescan. A stamped copy takes
    # the oldest unused target with the same stamp (or none); a different stamp never links; an unstamped copy takes
    # the oldest unused target of any stamp.
    host.turn(reply="ok")
    first, second = host.stamp(), host.stamp() + 1
    for stamp in (first, second, first):
        host.live.append({"role": "assistant", "content": "repeat", "timestamp": stamp})
    host.engine.ingest(host.live)
    targets = [r["store_id"] for r in host.engine._store.get_session_messages(SID) if r["content"] == "repeat"]
    assert len(targets) == 3
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    copies = [{"role": "assistant", "content": "repeat", "timestamp": second},
              {"role": "assistant", "content": "repeat", "timestamp": second + 50},
              {"role": "assistant", "content": "repeat"},
              {"role": "assistant", "content": "repeat", "timestamp": first}]
    sources = _ingest_copies(host, copies)
    assert len(sources) == 4
    assert _pairs(host.engine) == [(sources[0], targets[1]), (sources[2], targets[0]), (sources[3], targets[2])]
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 1


def test_25_a_rebound_conversation_never_spends_the_previous_conversations_rows(host):
    # #2335 review: a same-session rebind to another conversation. Its rows repeat the earlier conversation's reply,
    # but a replay target must share the copy's conversation as well as its session.
    host.turn(reply="ok")
    earlier = [r["store_id"] for r in host.engine._store.get_session_messages(SID) if r["content"] == "ok"]

    def rebind():
        host.engine.shutdown()
        host.engine = engine = LCMEngine(config=host.engine._config, hermes_home=str(host.tmp_path / "home"))
        host.engines.append(engine)
        engine.on_session_start(SID, platform="telegram", conversation_id="other-conversation", context_length=200_000)
        return engine

    engine = rebind()
    engine.ingest([{"role": "system", "content": "system prompt"},
                   {"role": "user", "content": "different head", "timestamp": host.stamp()},
                   {"role": "assistant", "content": "ok"}])
    later = [r["store_id"] for r in engine._store.get_session_messages(SID)
             if r["content"] == "ok" and r["store_id"] not in earlier]
    assert earlier and len(later) == 1  # both conversations stored their own reply
    assert not [pair for pair in _pairs(engine) if pair[1] in earlier]
    # A cursor-0 copy in the rebound conversation links to that conversation's row, never to the earlier one.
    engine = rebind()
    engine.ingest([{"role": "user", "content": "another head", "timestamp": host.stamp()},
                   {"role": "assistant", "content": "ok"}])
    _assert_cursor0(engine)
    source = engine._store.get_session_messages(SID)[-1]["store_id"]
    assert (source, later[0]) in _pairs(engine)
    assert not [pair for pair in _pairs(engine) if pair[1] in earlier]


@pytest.mark.parametrize("stored_id,links", [
    ("' ' || conversation_id || ' '", True),  # a padded id for the same conversation
    ("NULL", False),  # no conversation: it normalizes to '', which never matches a named one
])
def test_26_legacy_conversation_ids_link_only_within_their_conversation(host, stored_id, links):
    # #2335 review: legacy stores can hold a padded id for the same conversation. Ids are compared through the
    # store's normalizer, so those earlier rows remain targets; a blank or NULL id is not a wildcard (test_25).
    targets = _ok_candidates(host)
    conn = host.engine._store._conn
    conn.execute(f"UPDATE messages SET conversation_id = {stored_id} WHERE session_id = ?", (SID,))
    conn.commit()
    sources = _ingest_copies(host, [{"role": "assistant", "content": "ok"}] * 2)
    assert len(sources) == 2
    assert _pairs(host.engine) == (list(zip(sources, targets[:2])) if links else [])


def test_27_reasoning_strip_equals_the_regex_it_replaces():
    # #2335 review: the linear stripper must give exactly what _THINK_BLOCK_RE.sub gives, Unicode case folds included.
    import random
    from hermes_lcm import escalation

    pieces = ["<think>", "</think>", "<THINK >", "</Think\n>", "<thinking>", "</thinking>", "<thought>", "</thought>",
              "<reasoning>", "</reasoning>", "<reaſoning>", "</reaſoning>", "<REASONING_SCRATCHPAD>",
              "</reasoning_scratchpad>", "<thinK>", "<thİnk>", "</thınk>", "<", ">", "/", "a", " ", "\n",
              "<think", "think>", "</"]
    rng = random.Random(2305)
    for _ in range(20000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        want = escalation._THINK_BLOCK_RE.sub("", text) if text and "<" in text else text
        assert _strip_reasoning_blocks(text) == want, text


def test_28_unclosed_reasoning_tags_are_matched_once_per_spelling(monkeypatch):
    # #2335 review: the regex rescans to the end for every unclosed opening tag (16,000 took ~17 s under the ingest
    # write lock). A spelling with no closing tag is tried once; later openings with that spelling are skipped.
    from hermes_lcm import escalation

    calls = []

    class Counting:
        def match(self, text, pos):
            calls.append(pos)
            return block.match(text, pos)

    block = escalation._THINK_BLOCK_RE
    monkeypatch.setattr(escalation, "_THINK_BLOCK_RE", Counting())
    text = "<think>" * 16000 + "<thought>a</thought>" + "<THINK>" * 100 + "tail"
    assert _strip_reasoning_blocks(text) == "<think>" * 16000 + "<THINK>" * 100 + "tail"
    assert len(calls) == 2  # one failed <think>, one closed <thought>; <THINK> lowercases to the failed spelling
