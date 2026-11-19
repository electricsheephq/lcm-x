from copy import deepcopy

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_host_uid_shadow import _state_db


pytestmark = pytest.mark.issue(2962)


def _turn(uid, stamp, question="Q", answer="OK"):
    return [
        {"role": "user", "content": question, "message_uid": f"{uid}-q", "timestamp": stamp},
        {"role": "assistant", "content": answer, "message_uid": f"{uid}-a", "timestamp": stamp + 1},
    ]


def _open(path):
    config = LCMConfig(database_path=str(path / "lcm.db"), fresh_tail_count=2,
                       leaf_chunk_tokens=400, threshold_full_sweep_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(path))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _cold_ingest(path, stored, view):
    _state_db(path, [("S", None, None)])
    seed = _open(path)
    try:
        lineage, _ = seed._host_uid_lineage_key()
        assert lineage is not None
        bindings = []
        for message in stored:
            store_id = seed._store.append("S", message, conversation_id="conv")
            if uid := message.get("message_uid"):
                bindings.append((store_id, uid, "canonical", "stored_new"))
        seed._store.add_host_uid_bindings(lineage, bindings)
    finally:
        seed.shutdown()

    cold = _open(path)
    try:
        cold.ingest(deepcopy(view))
        rows = cold._store.get_session_messages("S")
        bindings = cold._store.host_uid_bindings_for(
            lineage, {message["message_uid"] for message in stored + view if "message_uid" in message},
        )
        return rows, bindings, cold._last_ingest_reconciliation["cursor"]
    finally:
        cold.shutdown()


def _assert_rows_and_bindings(rows, bindings, expected):
    assert [(r["role"], r["content"], r["observed_at"]) for r in rows] == [
        (m["role"], m["content"], m.get("timestamp")) for m in expected
    ]
    assert bindings == {
        m["message_uid"]: [(row["store_id"], "canonical")]
        for row, m in zip(rows, expected) if "message_uid" in m
    }


@pytest.mark.parametrize("overlap", [2, 0], ids=["A-two-turn-store", "B-one-turn-store"])
def test_cold_bind_preserves_fresh_repeated_turn(tmp_path, overlap):
    stored = _turn("s1", 1000.0) + (_turn("s2", 1002.0) if overlap else [])
    fresh = _turn("f", 2000.0) + _turn("next", 2002.0, "Next question", "Next answer")
    rows, bindings, cursor = _cold_ingest(tmp_path, stored, stored[len(stored) - overlap:] + fresh)

    assert (len(rows), cursor) == (len(stored) + len(fresh), overlap)
    _assert_rows_and_bindings(rows, bindings, stored + fresh)


def test_genuine_full_replay(tmp_path):
    stored = _turn("s1", 1000.0) + _turn("s2", 1002.0)
    rows, bindings, cursor = _cold_ingest(tmp_path, stored, stored)

    assert cursor == len(stored)
    _assert_rows_and_bindings(rows, bindings, stored)


@pytest.mark.parametrize("overlap", [2, 0], ids=["two-turn-store", "one-turn-store"])
def test_unkeyed_repeated_turn_retains_legacy_cursor(tmp_path, overlap):
    stored = _turn("s1", 1000.0) + (_turn("s2", 1002.0) if overlap else [])
    fresh = _turn("f", 2000.0) + _turn("next", 2002.0, "Next question", "Next answer")
    for message in stored + fresh:
        del message["timestamp"], message["message_uid"]
    rows, bindings, cursor = _cold_ingest(tmp_path, stored, stored[len(stored) - overlap:] + fresh)

    # #1566: without occurrence metadata this prefix is indistinguishable from full replay.
    assert cursor == len(stored)
    _assert_rows_and_bindings(rows, bindings, stored + fresh[2:])


def test_distinct_view_head(tmp_path):
    stored = _turn("s1", 1000.0) + _turn("s2", 1002.0)
    fresh = _turn("f", 2000.0, "Distinct question", "Distinct answer") + _turn("next", 2002.0)
    rows, bindings, cursor = _cold_ingest(tmp_path, stored, fresh)

    assert cursor == 0
    _assert_rows_and_bindings(rows, bindings, stored + fresh)
