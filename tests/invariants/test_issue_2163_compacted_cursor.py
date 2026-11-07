"""I1/I2: compacted history cannot prove a fresh reused-id tool occurrence."""

from copy import deepcopy

import pytest

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored
from tests.test_compression_boundary import _turn
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("dangling", [
    pytest.param(False, id="full-pair", marks=pytest.mark.xfail(strict=True, reason=(
        "#2172: a new call and result that repeat a pair compacted out of the view are taken as a replay"))),
    pytest.param(True, id="dangling-call"),
])
def test_compacted_away_identical_pair_is_not_proof(build, monkeypatch, dangling):
    def fixture(*, uids=False):
        rows = [row for i in range(2) for row in _turn(i)]
        rows += [{"role": "user", "content": "poll early"},
                 *tool_pair("call_0", "OK", tool_name="poll")]
        rows += [row for i in range(2, 12) for row in _turn(i)]
        for cid, text, stamp in (("older", "older lookup", 8),
                                 ("middle", "middle lookup", 9), ("c", "repeat", 10)):
            rows += [{"role": "user", "content": text, "timestamp": stamp},
                     *tool_pair(cid, shapes.LONG + cid, tool_name="lookup")]
        return rows

    monkeypatch.setattr(shapes, "fixture", fixture)
    engine, host = compact(build, mode="cold")
    assert not any(row.get("tool_call_id") == "call_0" for row in host)
    call, result = tool_pair("call_0", "OK", tool_name="poll")
    if dangling:
        host = deepcopy(host) + [{"role": "user", "content": "poll again"}, call]
        engine.ingest(deepcopy(host))
        engine.shutdown()
        engine = build()
        new = [result, {"role": "user", "content": "next"}]
    else:
        new = [{"role": "user", "content": "poll again"}, call, result,
               {"role": "user", "content": "next"}]
    before = stored(engine)
    engine.ingest(deepcopy(host) + new)
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [(row["role"], row["content"]) for row in added] == [
        (row["role"], row["content"]) for row in new
    ]
    assert added[-2]["tool_call_id"] == "call_0"


def _replay_links(engine):
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone():
        return {}
    return dict(conn.execute("SELECT store_id, related_store_id FROM message_relations "
                             "WHERE kind='possible_replay_of'"))


def _key(row):
    return row["role"], row.get("content"), row.get("tool_call_id"), row.get("tool_calls")


@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_cold_cursor_between_call_and_result_stores_no_silent_copy(build, monkeypatch, ledger):
    """#2163: a cold bind whose cursor lands between a stored call and its stored result.

    The soak saw the identity anchor stop there; the cursor is pinned to that boundary. The result
    at the cursor is a replay of the stored one: it is either not stored again or, when stored,
    linked ``possible_replay_of`` to the stored copy (visible, never silent). The new turn is stored.
    """
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    cursor = len(host) - 1
    assert host[cursor - 1]["role"] == "assistant" and host[cursor]["role"] == "tool"
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: cursor)
    new = {"role": "user", "content": "next"}
    engine.ingest(deepcopy(host) + [new])
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [_key(row) for row in added if row["role"] == "user"] == [_key(new)]
    links = _replay_links(engine)
    by_id = {row["store_id"]: row for row in before}
    silent = [row["store_id"] for row in added if row["role"] != "user"
              and _key(by_id.get(links.get(row["store_id"]), {"role": None})) != _key(row)]
    assert silent == []
