"""I1/I2/I8: stamped retry turns survive warm ingest and cold replay once (#1540)."""

from copy import deepcopy
import json
from typing import Any

import pytest

from tests.invariants.test_issue_1529_warm_proof_salvage import (
    Build, CLEARED, LONG, Row, build as build, compact, fixture, stored, assert_retry,
)
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("following", [2, 4])
@pytest.mark.parametrize("uid_change", ["all", "user-only"])
def test_same_stamp_known_uid_retry(build: Build, mode: str, following: int, uid_change: str) -> None:
    """A positive UID conflict vetoes replay even when the user stamp is equal."""
    engine, host = compact(build, mode=mode, uids=True)
    before = stored(engine)
    previous = deepcopy(host[-3:])
    for index, row in enumerate(host[-3:] if uid_change == "all" else host[-3:-2]):
        row["message_uid"] = f"{100 + index:032x}"
    assert host[-3]["timestamp"] == previous[0]["timestamp"]
    assert engine._host_occurrence_changed(host[-3], previous[0])
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
    assert counts == (1, 1, 1)
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
