"""I1/I2/I8: prefix edits and appended real results retain their occurrences (#1540)."""

from copy import deepcopy

import pytest

from tests.invariants.test_issue_1529_warm_proof_salvage import (
    CLEARED,
    LONG,
    build as build,
    fixture,
)
from tests.test_issue_671_stub_first_exit import tool_pair

U1, U2, PC, PR = [f"{n:032x}" for n in (0xA1, 0xA2, 0xB1, 0xB2)]


def rows(engine):
    return engine._store.get_session_messages(engine._session_id)


def anchor_switch(monkeypatch, anchor):
    if anchor == "off":
        monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "0")
    else:
        monkeypatch.delenv("LCM_IDENTITY_ANCHOR", raising=False)


def rotate(build, engine, original, mode):
    sid = engine._session_id
    engine.on_session_end(sid, deepcopy(original))
    engine.on_session_start(
        sid,
        boundary_reason="compression",
        old_session_id=sid,
        conversation_id=engine._conversation_id,
        platform="cli",
    )
    if mode == "cold":
        engine.shutdown()
        engine = build(restart=True)
    return engine


@pytest.mark.parametrize("edit", ["none", "content", "remove", "insert"])
@pytest.mark.parametrize("anchor", ["on", "off"])
@pytest.mark.parametrize("uid_case", ["new", "same"])
@pytest.mark.parametrize("pair_uids", [False, True])
@pytest.mark.parametrize("chrono", ["same-second", "late"])
@pytest.mark.parametrize("following", [0, 2])
def test_prefix_edit_user_uid_membership(
    build, monkeypatch, edit, anchor, uid_case, pair_uids, chrono, following
):
    anchor_switch(monkeypatch, anchor)
    engine = build()
    original = fixture(uids=False)
    original[-3]["message_uid"] = U1
    if pair_uids:
        (original[-2]["message_uid"], original[-1]["message_uid"]) = (PC, PR)
    earlier = []
    for k in range(6):
        earlier.append(
            {
                "role": "user",
                "content": f"after {k}",
                "timestamp": 10 if chrono == "same-second" else 11 + k,
            }
        )
        earlier.extend(tool_pair(f"after-{k}", LONG + f"after{k}", tool_name="lookup"))
        earlier.append({"role": "assistant", "content": f"answer after {k}"})
    original[-3:-3] = earlier
    engine.ingest(deepcopy(original))
    host = engine.compress(deepcopy(original), force=True)
    assert engine._last_compression_status == "compacted"
    assert host[-3:] == original[-3:]
    engine = rotate(build, engine, original, "cold")
    before = rows(engine)
    host = deepcopy(host)
    at = max(
        (
            i
            for (i, r) in enumerate(host[:-3])
            if r.get("role") == "user" and r.get("content") != "repeat"
        )
    )
    if edit == "content":
        host[at]["content"] += " PREFIX EDIT"
    elif edit == "remove":
        host.pop(at)
    elif edit == "insert":
        host.insert(
            at, {"role": "user", "content": "inserted earlier user", "timestamp": 7}
        )
    start = len(host) - 3
    host[start]["message_uid"] = U2 if uid_case == "new" else U1
    host[-1]["content"] = CLEARED
    for k in range(following):
        host.extend(tool_pair(f"following-{k}", f"new result {k}", tool_name="lookup"))
    host.append({"role": "assistant", "content": "final answer"})
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    assert rows(engine)[: len(before)] == before
    if uid_case == "new" and (not pair_uids):
        assert any(
            (
                r.get("tool_call_id") == "c" and r.get("content") == CLEARED
                for r in rows(engine)[len(before) :]
            )
        ), "known-new user lost its literal placeholder result"


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("anchor", ["on", "off"])
@pytest.mark.parametrize("stamp", ["unstamped", "stamped"])
@pytest.mark.parametrize("following", [0, 2])
def test_appended_result_cannot_inherit_prefix_proof(
    build, monkeypatch, mode, anchor, stamp, following
):
    from hermes_lcm.fresh_tail import tool_group_safe_end

    anchor_switch(monkeypatch, anchor)
    engine = build()
    original = fixture(uids=False)[:-1]
    original[-2]["message_uid"] = U1
    original[-1]["message_uid"] = PC
    assert original[-1].get("tool_calls")[0]["id"] == "c"
    assert tool_group_safe_end(original, len(original)) == len(original)
    engine.ingest(deepcopy(original))
    host = engine.compress(deepcopy(original), force=True)
    assert engine._last_compression_status == "compacted"
    call_at = next(
        (
            i
            for (i, r) in enumerate(host)
            if any((c.get("id") == "c" for c in r.get("tool_calls") or []))
        )
    )
    user_at = max(
        (i for (i, r) in enumerate(host[:call_at]) if r.get("role") == "user")
    )
    engine = rotate(build, engine, original, mode)
    before = rows(engine)
    assert not any((r.get("tool_call_id") == "c" for r in before))
    host = deepcopy(host)
    host[user_at]["message_uid"] = U2
    result = tool_pair("c", "unique appended result c", tool_name="lookup")[1]
    if stamp == "stamped":
        result["message_uid"] = PR
    host.append(result)
    for k in range(following):
        host.extend(
            [
                {"role": "user", "content": f"following turn {k}", "timestamp": 30 + k},
                {"role": "assistant", "content": f"following answer {k}"},
            ]
        )
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    assert rows(engine)[: len(before)] == before
    assert any(
        (
            r.get("tool_call_id") == "c"
            and r.get("content") == "unique appended result c"
            for r in rows(engine)
        )
    ), "appended real result was never stored"
