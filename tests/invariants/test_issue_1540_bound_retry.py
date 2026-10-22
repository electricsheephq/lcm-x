"""I1/I2/I8: stamped retry turns survive warm ingest and cold replay once (#1540)."""

from copy import deepcopy
import json
from typing import Any

import pytest

from tests.invariants.test_issue_1529_warm_proof_salvage import (
    Build, CLEARED, LONG, Row, build as build, compact, fixture, stored as raw_stored, assert_retry,
)
from tests.test_issue_671_stub_first_exit import tool_pair


def assert_complete_segments(rows: list[Row]) -> None:
    """Every result belongs to the active call, and each call has all results.

    A steady content update may reuse the last call occurrence across a user
    row (#1179); a later assistant call closes that segment permanently.
    """
    active: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        if row.get("role") == "assistant":
            assert active <= seen, ("split segment", active, seen)
            active = {call["id"] for call in row.get("tool_calls") or []}
            seen = set()
        elif row.get("role") == "tool":
            cid = row.get("tool_call_id")
            assert cid in active, ("orphan result", cid, active)
            seen.add(cid)
    assert active <= seen, ("incomplete segment", active, seen)


def stored(engine: Any) -> list[Row]:
    rows = raw_stored(engine)
    assert_complete_segments(rows)
    return rows


@pytest.mark.parametrize("mode,uids", [("cold", True), ("cold", False), ("warm", False)],
                         ids=["cold-user-uids", "cold-no-uids", "warm-control"])
def test_ambiguous_same_stamp_user_keeps_fresh_placeholder_pair(
    build: Build, monkeypatch: pytest.MonkeyPatch, mode: str, uids: bool,
) -> None:
    """A second matching user without an adjacent pair makes replay ambiguous."""
    import tests.invariants.test_issue_1529_warm_proof_salvage as shapes

    plain = shapes.fixture

    def two_users(*, uids: bool = False) -> list[Row]:
        rows = plain(uids=False)
        rows.append({"role": "user", "content": "repeat", "timestamp": 10})
        if uids:
            for index, row in enumerate(rows):
                if row["role"] == "user":
                    row["message_uid"] = f"{index + 1:032x}"
        return rows

    monkeypatch.setattr(shapes, "fixture", two_users)
    engine, host = compact(build, mode=mode, uids=uids)
    before = stored(engine)
    assert host[-1]["content"] == host[-4]["content"] == "repeat"
    assert host[-1]["timestamp"] == host[-4]["timestamp"] == 10
    if uids:
        assert host[-1]["message_uid"] != host[-4]["message_uid"]
    host.extend(tool_pair("c", CLEARED, tool_name="lookup"))
    for index in range(2):
        host.extend(tool_pair(f"following-{index}", f"new result {index}", tool_name="lookup"))
    host.append({"role": "assistant", "content": "final answer"})
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    counts = (sum(any(call["id"] == "c" for call in row.get("tool_calls") or []) for row in after),
              sum(row.get("tool_call_id") == "c" for row in after))
    assert counts == (2, 2), "fresh pair after the cursor-matched user was suppressed"
    assert len(after) - len(before) == 7


@pytest.mark.parametrize("restart", [False, True], ids=["steady", "restart-after-15"])
def test_replay_segment_helper_only_runs_on_reconcile_ingest(build: Build, monkeypatch, restart: bool) -> None:
    """Thirty steady turns include compaction; only a restart invokes the helper."""
    from hermes_lcm.reconcile import ReconcileMixin

    original = ReconcileMixin._replayed_tool_segment_indexes_after_cursor
    calls = []

    def counted(self, messages, cursor):
        calls.append(cursor)
        return original(self, messages, cursor)

    monkeypatch.setattr(ReconcileMixin, "_replayed_tool_segment_indexes_after_cursor", counted)
    engine = build()
    host = []
    per_turn = []
    for turn in range(1, 31):
        if restart and turn == 16:
            engine.shutdown()
            engine = build(restart=True)
        host.append({"role": "user", "content": f"turn {turn} " + LONG, "timestamp": turn})
        if turn % 2 == 0:
            host.extend(tool_pair(f"tool-{turn}", LONG + str(turn), tool_name="lookup"))
        host.append({"role": "assistant", "content": f"answer {turn} " + LONG})
        before = len(calls)
        engine.ingest(deepcopy(host))
        stored(engine)
        if turn == 10:
            host = engine.compress(deepcopy(host), force=True)
            assert engine._last_compression_status == "compacted"
        per_turn.append(len(calls) - before)
    assert per_turn == [int(restart and turn == 16) for turn in range(1, 31)]
    assert all(cursor > 0 for cursor in calls)


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("following", [2, 4])
@pytest.mark.parametrize("uid_change", ["all", "user-only", "pair-only"])
def test_same_stamp_known_uid_retry(build: Build, mode: str, following: int, uid_change: str) -> None:
    """Each known UID identifies its own row, independently of the other rows."""
    engine, host = compact(build, mode=mode, uids=True)
    before = stored(engine)
    previous = deepcopy(host[-3:])
    changed = host[-3:] if uid_change == "all" else host[-3:-2] if uid_change == "user-only" else host[-2:]
    for index, row in enumerate(changed):
        row["message_uid"] = f"{100 + index:032x}"
    assert host[-3]["timestamp"] == previous[0]["timestamp"]
    assert engine._host_occurrence_changed(host[-3], previous[0]) == (uid_change != "pair-only")
    if uid_change == "user-only":
        assert [row["message_uid"] for row in host[-2:]] == [row["message_uid"] for row in previous[1:]]
    host[-1]["content"] = CLEARED
    for index in range(following):
        host.extend(tool_pair(f"following-{index}", f"new result {index}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    counts = (sum(row["role"] == "user" and row["content"] == "repeat" for row in added),
              sum(any(call["id"] == "c" for call in row.get("tool_calls") or []) for row in added),
              sum(row.get("tool_call_id") == "c" and row["content"] == CLEARED for row in added))
    engine.ingest(deepcopy(host))
    repeated = stored(engine)
    print("\nUID_CASE " + json.dumps(dict(mode=mode, following=following, uid_change=uid_change,
                                          counts=counts, repeat_added=len(repeated) - len(after))))
    assert counts == {"all": (1, 1, 1), "user-only": (1, 0, 0), "pair-only": (0, 1, 1)}[uid_change]
    assert repeated == after


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("following", [2, 4])
def test_retry_with_following_tools(build: Build, mode: str, following: int) -> None:
    """Later results can permit salvage, but cannot erase a new turn's pair."""
    engine, host = compact(build, mode=mode)
    before = stored(engine)
    host[-3]["timestamp"] = 20
    host[-1]["content"] = CLEARED
    pair: Any = tool_pair
    for index in range(following):
        host.extend(pair(f"following-{index}", f"new result {index}", tool_name="lookup"))
    assert_retry(engine, host, before, CLEARED)
    after = stored(engine)
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


@pytest.mark.parametrize("mode", ["warm", "cold"])
def test_retry_invalidation_stops_at_next_user(build: Build, mode: str) -> None:
    engine, host = compact(build, mode=mode)
    before = stored(engine)
    next(row for row in host if row.get("content") == "middle lookup")["timestamp"] = 20
    for row in host:
        if row.get("tool_call_id") in {"middle", "c"}:
            row["content"] = CLEARED
    pair: Any = tool_pair
    for index in range(2):
        host.extend(pair(f"following-{index}", f"new result {index}", tool_name="lookup"))
    engine.ingest(host)
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert sum(row["role"] == "user" for row in added) == 1
    assert sum(any(call["id"] == "middle" for call in row.get("tool_calls") or []) for row in added) == 1
    assert sum(row.get("tool_call_id") == "middle" and row["content"] == CLEARED for row in added) == 1
    assert not any(row.get("tool_call_id") == "c" or any(
        call["id"] == "c" for call in row.get("tool_calls") or []) for row in added)
    assert len(added) == 7
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


@pytest.mark.parametrize("initial", [LONG + "c", CLEARED], ids=["long", "literal-placeholder"])
@pytest.mark.parametrize("change", [False, True], ids=["unchanged", "changed"])
def test_steady_retry_literal_control(build: Build, initial: str, change: bool) -> None:
    engine = build()
    host = fixture()[-3:]
    host[-1]["content"] = initial
    engine.ingest(deepcopy(host))
    before = stored(engine)
    host[0]["timestamp"] = 20
    if change:
        host[-1]["content"] = CLEARED if initial != CLEARED else "new genuine output"
    engine.ingest(deepcopy(host))
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert sum(row["role"] == "user" for row in added) == 1
    assert sum(bool(row.get("tool_calls")) for row in added) == 0
    assert sum(row["role"] == "tool" for row in added) == int(change)
    if change:
        assert added[-1]["content"] == host[-1]["content"]
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("following", [2, 4])
@pytest.mark.parametrize("clear_earlier", [False, True], ids=["plain", "earlier-salvage"])
def test_restart_keeps_retry_once(build: Build, mode: str, following: int, clear_earlier: bool) -> None:
    """I1/I2/I8: a literal retry pair remains durable across restart (#1540)."""
    def counts(rows: list[Row]) -> tuple[int, int]:
        return (
            sum(any(call["id"] == "c" for call in row.get("tool_calls") or []) for row in rows),
            sum(row.get("tool_call_id") == "c" and row["content"] == CLEARED for row in rows),
        )

    engine, host = compact(build, mode=mode)
    original = stored(engine)
    assert counts(original) == (1, 0)
    host[-3]["timestamp"] = 20
    host[-1]["content"] = CLEARED
    pair: Any = tool_pair
    for index in range(following):
        host.extend(pair(f"following-{index}", f"new result {index}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    first = stored(engine)
    assert first[:len(original)] == original
    assert counts(first) == (2, 1)
    assert len(first) == len(original) + 3 + 2 * following
    engine.shutdown()
    engine = build()
    assert stored(engine) == first
    if clear_earlier:
        next(row for row in host if row.get("tool_call_id") == "older")["content"] = CLEARED
    host.append({"role": "user", "content": "after restart", "timestamp": 50})
    engine.ingest(deepcopy(host))
    final = stored(engine)
    assert final[:len(first)] == first
    assert counts(final) == (2, 1)
    assert len(final) == len(first) + 1
    assert final[-1]["content"] == "after restart"
    engine.ingest(deepcopy(host))
    assert stored(engine) == final



@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("with_user", ["new-stamp-user", "no-user", "uid-user"])
def test_fresh_literal_after_cold_restart(build, mode, with_user):
    engine, host = compact(build, mode=mode)
    engine.shutdown()
    engine = build()
    before = stored(engine)
    if with_user == "new-stamp-user":
        host.append({"role": "user", "content": "repeat", "timestamp": 60})
    elif with_user == "uid-user":
        host.append({"role": "user", "content": "repeat", "message_uid": "f" * 32})
    host.extend(tool_pair("c", CLEARED, tool_name="lookup"))  # reused id, genuinely literal placeholder
    for i in range(2):
        host.extend(tool_pair(f"later-{i}", f"later result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    added = stored(engine)[len(before):]
    got = (sum(r["role"] == "user" for r in added),
           sum(any(c["id"] == "c" for c in r.get("tool_calls") or []) for r in added),
           sum(r.get("tool_call_id") == "c" and r["content"] == CLEARED for r in added),
           len(added))
    print(f"\nATTACK {mode} {with_user} added(user,call,result,total)={got}")
    assert got == (0 if with_user == "no-user" else 1, 1, 1, (1 if with_user != "no-user" else 0) + 2 + 4)
    n = len(stored(engine))
    engine.ingest(deepcopy(host))
    assert len(stored(engine)) == n


@pytest.mark.parametrize("mode", ["warm", "cold"])
def test_unstamped_user_retry_control(build, mode):
    """Gateway control: retried user row without a known stamp change cannot be told from a replay."""
    engine, host = compact(build, mode=mode)
    before = stored(engine)
    host[-1]["content"] = CLEARED
    for i in range(2):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    added = stored(engine)[len(before):]
    print(f"\nCONTROL unstamped {mode} added={len(added)} c_rows={sum(r.get('tool_call_id') == 'c' for r in added)}")
    assert len(added) == 4 and not any(r.get("tool_call_id") == "c" for r in added)


@pytest.mark.parametrize("mode", ["warm", "cold"])
def test_stamped_undo_pair_is_fresh(build: Build, mode: str) -> None:
    engine, _host = compact(build, mode=mode)
    engine.shutdown()
    engine = build()
    engine.ingest([])  # /undo emptied the host view; the store retains its history.
    before = stored(engine)
    host = [{"role": "user", "content": "repeat", "timestamp": 60},
            *tool_pair("c", CLEARED, tool_name="lookup")]
    for index, row in enumerate(host[1:], 61):
        row["timestamp"] = index
    engine.ingest(deepcopy(host))
    added = stored(engine)[len(before):]
    counts = (sum(row["role"] == "user" for row in added),
              sum(bool(row.get("tool_calls")) for row in added),
              sum(row.get("tool_call_id") == "c" for row in added))
    assert counts == (1, 1, 1)
    after = stored(engine)
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("uid_change", ["all", "pair-only", "equal"])
@pytest.mark.parametrize("prefix_change", ["edit", "insert"])
def test_prefix_change_requires_positive_uid_proof(build, mode, uid_change, prefix_change):
    engine, host = compact(build, mode=mode, uids=True)
    before = stored(engine)
    older = next(i for i, row in enumerate(host) if row.get("content") == "middle lookup")
    if prefix_change == "edit":
        host[older]["content"] += " edited"
    else:
        host.insert(older, {"role": "user", "content": "inserted prefix"})
    if uid_change != "equal":
        for i, row in enumerate(host[-3:] if uid_change == "all" else host[-2:]):
            row["message_uid"] = f"{200 + i:032x}"
    host[-1]["content"] = CLEARED
    for i in range(2):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    after = stored(engine)
    added = after[len(before):]
    counts = (sum(bool(r.get("tool_calls")) and r["tool_calls"][0]["id"] == "c" for r in added),
              sum(r.get("tool_call_id") == "c" for r in added))
    print("B1", mode, uid_change, prefix_change, counts)
    assert counts == ((0, 0) if uid_change == "equal" else (1, 1))
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


def parallel_compact(build, mode, uids):
    engine = build()
    original = fixture(uids=uids)
    other_call, other_result = tool_pair("d", LONG + "d", tool_name="lookup")
    original[-2]["tool_calls"].extend(other_call["tool_calls"])
    if uids:
        other_result["message_uid"] = f"{90:032x}"
    original.append(other_result)
    engine.ingest(deepcopy(original))
    output = engine.compress(deepcopy(original), force=True)
    assert engine._last_compression_status == "compacted"
    sid = engine._session_id
    engine.on_session_end(sid, deepcopy(original))
    engine.on_session_start(sid, boundary_reason="compression", old_session_id=sid,
                            conversation_id=engine._conversation_id, platform="cli")
    if mode == "cold":
        engine.shutdown()
        engine = build(restart=True)
    return engine, output


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("new_user", [False, True])
def test_reordered_parallel_retry_keeps_new_uid_results(build, mode, new_user):
    engine, host = parallel_compact(build, mode, True)
    before = stored(engine)
    for i, row in enumerate(host[-4:] if new_user else host[-3:]):
        row["message_uid"] = f"{300 + i:032x}"
    host[-2]["content"] = CLEARED
    host[-2:] = reversed(host[-2:])
    for i in range(2):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    after = stored(engine)
    added = after[len(before):]
    assert sum(r.get("tool_call_id") == "c" for r in added) == 1
    assert sum(r.get("tool_call_id") == "d" for r in added) == 1
    assert [r.get("tool_call_id") for r in added if r.get("tool_call_id") in {"c", "d"}] == ["d", "c"]
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("changed_output", [False, True])
def test_parallel_borrowed_retry_is_atomic(build, mode, reorder, changed_output):
    engine, host = parallel_compact(build, mode, False)
    before = stored(engine)
    host[-2]["content"] = CLEARED
    if changed_output:
        host[-1]["content"] = "new d output"
    if reorder:
        host[-2:] = reversed(host[-2:])
    for i in range(2):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    after = stored(engine)
    added = after[len(before):]
    counts = tuple(sum((any(c["id"] == "c" for c in r.get("tool_calls") or []) if kind == "call"
                        else r.get("tool_call_id") == kind) for r in added) for kind in ("call", "c", "d"))
    print("B3", mode, reorder, changed_output, counts)
    assert counts in ((0, 0, 0), (1, 1, 1))
    if changed_output:
        assert counts == (1, 1, 1)
    engine.ingest(deepcopy(host))
    assert stored(engine) == after
    engine.shutdown()
    engine = build()
    engine.ingest(deepcopy(host))
    restarted = stored(engine)
    delta = restarted[len(after):]
    assert sum(r.get("tool_call_id") == "c" for r in delta) <= 1
    assert sum(r.get("tool_call_id") == "d" for r in delta) <= 1
    engine.ingest(deepcopy(host))
    assert stored(engine) == restarted


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("following", [2, 4])
@pytest.mark.parametrize("shape", ["same-stamp", "new-stamp", "prefix-edit"])
@pytest.mark.parametrize("clear_earlier", [False, True])
def test_missing_uid_compatibility(build, mode, following, shape, clear_earlier):
    engine, host = compact(build, mode=mode)
    before = stored(engine)
    assert not any(r.get("message_uid") for r in host)
    if shape == "new-stamp":
        host[-3]["timestamp"] = 20
    elif shape == "prefix-edit":
        next(r for r in host if r.get("content") == "middle lookup")["content"] += " edited"
    if clear_earlier:
        next(r for r in host if r.get("tool_call_id") == "older")["content"] = CLEARED
    host[-1]["content"] = CLEARED
    for i in range(following):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    after = stored(engine)
    added = after[len(before):]
    counts = (sum(r.get("role") == "user" for r in added),
              sum(bool(r.get("tool_calls")) for r in added),
              sum(r.get("role") == "tool" for r in added))
    print("MISSING_UID", mode, following, shape, clear_earlier, counts)
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize(("metadata", "user_change"), [
    ("call-only-uid", "new-uid"), ("result-only-uid", "new-uid"),
    ("call-only-uid", "new-stamp"), ("result-only-uid", "new-stamp"),
    ("call-only-stamp", "new-stamp"), ("result-only-stamp", "new-stamp"),
])
def test_mixed_metadata_retry_keeps_segment_atomic(build, monkeypatch, mode, metadata, user_change):
    """Known segment proof suppresses unknown siblings without splitting (#1830 r9)."""
    import tests.invariants.test_issue_1529_warm_proof_salvage as shapes

    plain = shapes.fixture
    is_uid = metadata.endswith("uid")
    known_index = -2 if metadata.startswith("call-only") else -1

    def mixed_fixture(*, uids=False):
        rows = plain(uids=uids)
        if is_uid:
            rows[-1 if known_index == -2 else -2].pop("message_uid", None)
        else:
            rows[known_index]["timestamp"] = 11
        return rows

    monkeypatch.setattr(shapes, "fixture", mixed_fixture)
    engine, host = compact(build, mode=mode, uids=is_uid)
    before = stored(engine)
    if user_change == "new-uid":
        host[-3]["message_uid"] = f"{500:032x}"
    else:
        host[-3]["timestamp"] = 20
    host[-1]["content"] = CLEARED
    for i in range(2):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    after = stored(engine)  # Full-session no-lone-call/result invariant, warm AND cold.
    assert after[:len(before)] == before
    added = after[len(before):]
    counts = (sum(any(c["id"] == "c" for c in row.get("tool_calls") or []) for row in added),
              sum(row.get("tool_call_id") == "c" for row in added))
    if mode == "warm":
        assert counts == (0, 0)  # Exact base parity in all six warm shapes.
    else:
        assert counts[0] == counts[1]  # Base's lone result (0/1) must be fixed.
    assert sum(row.get("content") == "repeat" for row in added) == 1
    for i in range(2):
        assert sum(row.get("tool_call_id") == f"following-{i}" for row in added) == 1
    print("R9_MIXED", mode, metadata, user_change, counts)
    engine.ingest(deepcopy(host))
    assert stored(engine) == after


@pytest.mark.parametrize("fault", ["corrupt-proof", "remap-raises"])
def test_replay_proof_failure_matches_base(build, monkeypatch, fault):
    """Optional proof faults retain the six complete rows stored by base 9eacedb9."""
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    next(row for row in host if row.get("content") == "middle lookup")["timestamp"] = 20
    host[-1]["content"] = CLEARED
    for i in range(2):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    helper = engine._replayed_tool_segment_indexes_after_cursor
    read = engine._store.read_metadata_json
    calls = []
    faults = []

    def corrupt_read(key):
        if "compaction_commit_proof" in key:
            faults.append(key)
            return json.loads("{invalid JSON")
        return read(key)

    def broken_remap(*args, **kwargs):
        faults.append("remap")
        raise RuntimeError("synthetic remap failure")

    def faulted(messages, cursor):
        calls.append(cursor)
        # Fault the helper's own read, after the other readers have reconciled.
        with monkeypatch.context() as patch:
            if fault == "corrupt-proof":
                patch.setattr(engine._store, "read_metadata_json", corrupt_read)
            else:
                patch.setattr(engine, "_remapped_output_metadata", broken_remap)
            replayed = helper(messages, cursor)
        return replayed

    monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", faulted)
    engine.ingest(deepcopy(host))
    assert len(calls) == 1 and calls[0] > 0
    assert faults, "the fault must reach the optional proof reader"
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    def project(row):
        return {key: row.get(key) for key in ("role", "content", "tool_calls", "tool_call_id")}
    # Frozen from base 9eacedb9 cold corrupt-proof/remap-raises probe outputs.
    expected = [
        {
            "role": "user",
            "content": "middle lookup",
            "tool_calls": None,
            "tool_call_id": None
        },
        {
            "role": "tool",
            "content": "[Old tool output cleared to save context space]",
            "tool_calls": None,
            "tool_call_id": "c"
        },
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": "following-0",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{}"
                    }
                }
            ],
            "tool_call_id": None
        },
        {
            "role": "tool",
            "content": "new result 0",
            "tool_calls": None,
            "tool_call_id": "following-0"
        },
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": "following-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{}"
                    }
                }
            ],
            "tool_call_id": None
        },
        {
            "role": "tool",
            "content": "new result 1",
            "tool_calls": None,
            "tool_call_id": "following-1"
        }
    ]
    assert [project(row) for row in after[len(before):]] == expected
    assert_complete_segments(after)


@pytest.mark.parametrize("uids", [False, True], ids=["no-uids", "stored-uids"])
@pytest.mark.parametrize("following", [2, 4])
def test_fresh_same_stamp_user_ordinal_keeps_pair(build, uids, following):
    """An appended U2 cannot borrow U1's pair while U1 remains on the host."""
    engine, host = compact(build, mode="cold", uids=uids)
    before = stored(engine)
    host.extend([{"role": "user", "content": "repeat", "timestamp": 10},
                 *tool_pair("c", CLEARED, tool_name="lookup")])
    for i in range(following):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    host.append({"role": "assistant", "content": "final answer"})
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    counts = (sum(r["role"] == "user" and r["content"] == "repeat" for r in after),
              sum(any(c["id"] == "c" for c in r.get("tool_calls") or []) for r in after),
              sum(r.get("tool_call_id") == "c" for r in after))
    assert counts == (2, 2, 2), "fresh same-stamp occurrence lost its pair"


@pytest.mark.parametrize("form", ["append", "replace"])
@pytest.mark.parametrize("uid_case", ["different", "same", "adopt"])
def test_user_uid_veto_without_pair_uids(build, monkeypatch, form, uid_case):
    """A known-different user vetoes binding; absent stored UIDs allow adoption."""
    import tests.invariants.test_issue_1529_warm_proof_salvage as shapes

    plain = shapes.fixture
    u1, u2 = f"{0xa1:032x}", f"{0xa2:032x}"

    def user_uid_only(*, uids=False):
        rows = plain(uids=False)
        if uid_case != "adopt":
            rows[-3]["message_uid"] = u1
        return rows

    monkeypatch.setattr(shapes, "fixture", user_uid_only)
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    uid = u1 if uid_case == "same" else u2
    if form == "append":
        host.extend([{"role": "user", "content": "repeat", "timestamp": 10, "message_uid": uid},
                     *tool_pair("c", CLEARED, tool_name="lookup")])
    else:
        host[-3]["message_uid"] = uid
        host[-1]["content"] = CLEARED
    for i in range(2):
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    counts = (sum(any(c["id"] == "c" for c in r.get("tool_calls") or []) for r in added),
              sum(r.get("tool_call_id") == "c" for r in added))
    # Only replacement with an equal or adopted UID is a proven replay.
    expected = (0, 0) if form == "replace" and uid_case != "different" else (1, 1)
    assert counts == expected, "new user UID must not suppress an unlabelled pair"


def test_ambiguous_user_rejects_first_and_last_match(build, monkeypatch):
    """The replay scan cannot arbitrarily choose either matching user's pair."""
    import tests.invariants.test_issue_1529_warm_proof_salvage as shapes

    plain = shapes.fixture

    def repeated_pair_fixture(*, uids=False):
        rows = plain(uids=False)
        rows[-3:-3] = deepcopy(rows[-3:])
        return rows

    monkeypatch.setattr(shapes, "fixture", repeated_pair_fixture)
    engine, _host = compact(build, mode="cold")
    # Exercise both positions explicitly: cold cursor reconciliation may skip
    # the last pair before the replay helper, hiding a last-match mutation.
    for occurrence in (0, 1):
        host = repeated_pair_fixture()
        user_index = len(host) - 6 + 3 * occurrence
        host[user_index + 2]["content"] = CLEARED
        for i in range(2):
            host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))
        replayed = engine._replayed_tool_segment_indexes_after_cursor(host, user_index + 1)
        assert user_index + 1 not in replayed, "ambiguous first/last match suppressed the call"
        assert user_index + 2 not in replayed, "ambiguous first/last match suppressed the result"
