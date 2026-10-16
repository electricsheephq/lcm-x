"""#659 / PR #1073: optional carry robustness, red on c5432b92."""

import logging
import math
import sqlite3
from collections import Counter

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.db_bootstrap import refresh_legacy_conversation_ids
from hermes_lcm.tokens import count_messages_tokens
import tests.test_issue_659_user_carry_packet as carry_packet
from tests.test_issue_659_user_carry_packet import CARRY, SEP, _compact, _generated, _history, _status

make = carry_packet.make


def _ending():
    return [{"role": "user", "content": "latest request"}, {"role": "assistant", "content": "ok"}]


@pytest.mark.parametrize("ids", ["1, ", "1,2", "9" * 5000], ids=["trailing-comma", "missing-space", "huge-int"])
def test_f1_malformed_ids_are_carried_as_ordinary_content(make, ids):
    engine = make()
    pasted = SEP + CARRY + f" (store ids {ids})\nOrdinary pasted user wording."
    messages = _history(8)
    messages[0]["content"] = pasted
    out = _compact(engine, [*messages, *_ending()])
    assert out and pasted in _generated(engine, out)[1]
    rows = engine._store.get_session_messages("S")
    stored = next(row for row in rows if row["content"] == pasted)
    assert not engine._holds_verified_user_carry(stored)
    assert engine._verified_generated_suffix_end(pasted, 0) == 0
    again = engine._assemble_context(None, out[1:])
    assert again and pasted in _generated(engine, again)[1]


def test_f2_model_window_change_rebuilds_packet(make):
    engine = make(window=128_000)
    messages = _history(12)
    for row in messages:
        if row["role"] == "user":
            row["content"] += "Earlier user detail. " * 300
    out = _compact(engine, [*messages, *_ending()])
    old_cache = engine._user_carry_cache
    old_budget = _status(engine)["budget_tokens"]
    engine.update_model("synthetic-smaller-window", context_length=32_000)
    again = engine._assemble_context(None, out[1:])
    status = _status(engine)
    assert engine._user_carry_cache is not old_cache
    assert 0 < status["budget_tokens"] < old_budget
    assert status["budget_tokens"] <= int(engine.context_length * 0.15)
    assert 0 < status["delivered_tokens"] <= status["budget_tokens"]
    assert _generated(engine, again)[1]
    assert count_messages_tokens(again) <= engine.threshold_tokens


@pytest.mark.parametrize("failure", ["page", "lineage", "cached"])
def test_f3_optional_failure_keeps_compaction_and_assembly(make, monkeypatch, caplog, failure):
    engine = make()
    out = _compact(engine, [*_history(8), *_ending()])
    assert _generated(engine, out)[1] and engine._user_carry_cache is not None

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    if failure == "page":
        engine._user_carry_cache = None
        monkeypatch.setattr(engine._store, "load_lineage_user_rows", locked)
    elif failure == "lineage":
        monkeypatch.setattr(engine, "_user_carry_lineage", locked)
    else:
        original = engine_module.count_message_tokens

        def fail_carry_tokens(message):
            if message.get("content", "").startswith(CARRY):
                locked()
            return original(message)

        monkeypatch.setattr(engine_module, "count_message_tokens", fail_carry_tokens)

    with caplog.at_level(logging.WARNING, logger=engine_module.__name__):
        again = engine._assemble_context(None, out[1:])
    assert again and all(CARRY not in row.get("content", "") for row in again)
    assert all(engine_module._OMITTED_SUMMARIES_PREFIX not in row.get("content", "") for row in again)
    assert engine._user_carry_cache is None
    assert _status(engine) == {"error": "OperationalError", "empty_no_room": False, "delivered_tokens": 0}
    warnings = [record for record in caplog.records if "LCM user carry failed" in record.getMessage()]
    assert len(warnings) == 1
    caplog.clear()
    compressed = _compact(engine, [*_history(8), *_ending()])
    assert compressed and all(CARRY not in row.get("content", "") for row in compressed)
    assert _status(engine)["error"] == "OperationalError"
    assert engine._user_carry_cache is None


@pytest.mark.parametrize("legacy_id", [" conv ", " \t "])
def test_f4_legacy_ownership_verifies_packet_and_prevents_nesting(make, legacy_id):
    """A legacy (padded or blank) conversation id: selection owns the row and verification accepts it, because
    the store normalizes ids on read (_row_to_dict). Guards against either side drifting from the other."""
    engine = make()
    out = _compact(engine, [*_history(8), *_ending()])
    conn = engine._store.connection
    row = next(row for row in engine._store.get_session_messages("S") if row["role"] == "user")
    conn.execute("UPDATE messages SET conversation_id = ? WHERE store_id = ?", (legacy_id, row["store_id"]))
    conn.commit()
    refresh_legacy_conversation_ids(conn)
    engine._user_carry_cache = None
    selected = engine._store.load_lineage_user_rows(["S"], "conv", engine._last_compacted_store_id + 1, 200)
    assert row["store_id"] in {item["store_id"] for item in selected}
    again = engine._assemble_context(None, out[1:])
    carry = _generated(engine, again)[1]
    assert row["content"] in carry
    assert engine._verified_generated_suffix_end(SEP + carry, 0) == len(SEP + carry)
    pure = again[0]["content"][:engine._verified_lcm_summary_prefix_end(again[0]["content"])]
    assert CARRY in pure and engine._holds_verified_user_carry({"role": "user", "content": pure})
    replay = _compact(engine, [*again, *_history(4), *_ending()])
    assert replay and _generated(engine, replay)[1].count(CARRY) == 1


def test_f5_ineligible_rows_bound_each_scan(make, monkeypatch):
    engine = make()
    out = _compact(engine, [*_history(8), *_ending()])
    assert "scan_capped" not in _status(engine)
    # Blank stored user rows are ineligible. Put them before eligible rows so
    # the newest scan exhausts its eligible rows and the origin scan finds none.
    rows = [{"role": "user", "content": " "}] * 2500 + [
        {"role": "user", "content": f"Eligible extra user row {i}."} for i in range(3)]
    ids = engine._store.append_batch("S", rows, conversation_id="conv")
    engine._last_compacted_store_id = ids[-1]
    engine._user_carry_cache = None
    calls = Counter()
    original = engine._store.load_lineage_user_rows

    def counted(*args, newest_first=True, **kwargs):
        calls[newest_first] += 1
        return original(*args, newest_first=newest_first, **kwargs)

    monkeypatch.setattr(engine._store, "load_lineage_user_rows", counted)
    again = engine._assemble_context(None, out[1:])
    assert again and _status(engine)["scan_capped"] is True
    assert set(calls) == {True, False}
    assert all(count <= math.ceil(2000 / 200) + 1 for count in calls.values()), calls
    assert _compact(engine, [*again, *_history(4), *_ending()])


@pytest.mark.parametrize("newest", [True, False])
def test_f6_large_lineage_uses_one_bind_and_preserves_query(make, newest):
    engine = make()
    rows = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "excluded role"},
            {"role": "user", "content": "second"}, {"role": "user", "content": "third"}]
    ids = engine._store.append_batch("S", rows, conversation_id="conv")
    engine._store.append("S", {"role": "user", "content": "excluded conversation"}, conversation_id="other")
    lineage = ["S", *(f"synthetic-{i}" for i in range(39_999))]
    conn = engine._store.connection
    previous = conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 32_766)
    try:
        cursor = ids[-1] if newest else ids[0]
        found = engine._store.load_lineage_user_rows(lineage, "conv", cursor, 1, newest_first=newest)
        assert [row["content"] for row in found] == ["second"]
        all_rows = engine._store.load_lineage_user_rows(lineage, "conv", ids[-1] + 1 if newest else 0, 200,
                                                       newest_first=newest)
        assert [row["content"] for row in all_rows] == (["third", "second", "first"] if newest else
                                                       ["first", "second", "third"])
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)
