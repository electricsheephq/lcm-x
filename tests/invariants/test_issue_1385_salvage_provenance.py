"""I1/I2/I8: a placeholder must not borrow an impossible salvage source (#1385)."""

import copy
import json

import pytest

from tests.harness import engine_context, stored_rows
from tests.test_issue_773_hermes_pruned_tool_replay import (
    CLEARED,
    SYSTEM,
    _call,
    _grouped,
    _rebind,
    _result,
    make_engine,  # noqa: F401
    plain_engine,
    stored,
)


@pytest.mark.parametrize("matcher", ["host-rewrite", "commit-proof"])
def test_legacy_placeholder_copy_with_host_rewrite_replays_once(make_engine, matcher):  # noqa: F811
    """I1/I2/I8: an older literal copy also admits its salvage alias on replay.

    Two newer results let the host borrow X's full identity; the rewritten final
    user row prevents an ordinary literal-tail match from hiding the mismatch.
    """
    first = plain_engine(make_engine, fresh_tail_count=64)
    original = [SYSTEM, {"role": "user", "content": "original turn"},
                _call("X"), _result("X", "X" * 300)]
    first._ingest_messages(original)
    if matcher == "commit-proof":
        identities = [first._proof_replay_identity(row) for row in original]
        first._persist_compress_commit_proof({
            **first._emission_binding(), "session_id": first._session_id,
            "conversation_id": first._conversation_id,
            "output": identities, "output_effective": identities, "emissions": [],
        })
        assert first._durable_commit_proof_payload() is not None
    legacy = [_call("X"), _result("X", CLEARED),
              _call("Y"), _result("Y", "result Y"),
              _call("Z"), _result("Z", "result Z"),
              {"role": "user", "content": "later turn \n"}]
    for row in legacy:  # persisted by a legacy writer, bypassing current reconciliation
        first._store.append(first._session_id, row, conversation_id=first._conversation_id)
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    before = stored(engine)
    host = copy.deepcopy(original + legacy)
    host[-1]["content"] = "later turn"
    host.append({"role": "user", "content": "next turn"})
    cursor = (engine._cursor_from_host_rewrite_head(host, len(before)) if matcher == "host-rewrite"
              else engine._cursor_from_durable_commit_proof(host))

    engine._ingest_messages(host)

    rows = stored(engine)
    assert rows[:-1] == before  # no duplicate group, no chronology or durable-row changes
    assert rows[-1]["content"] == "next turn"
    assert cursor == len(before)
    assert _grouped(rows)
    engine._ingest_messages(copy.deepcopy(host))
    assert stored(engine) == rows


@pytest.mark.parametrize("stored_content,replayed_content,already_durable", [
    (CLEARED, CLEARED, True),
    ("X" * 300, CLEARED, True),
    (CLEARED, CLEARED + " different literal", False),
    (CLEARED, "genuinely new result with the same id", False),
], ids=["literal", "alias", "different-literal", "new-same-id"])
def test_middle_window_literal_placeholder_replay(tmp_path, stored_content, replayed_content, already_durable):
    """#1437, I1/I2/I10: exact literal and proven alias forms share the suffix rule."""
    options = dict(window=200_000, fresh_tail_count=64,
                   large_output_externalization_enabled=False,
                   large_output_active_replay_stubbing_enabled=False)
    original = [SYSTEM, {"role": "user", "content": "original turn"},
                _call("X"), _result("X", "X" * 300)]
    legacy = [_call("X"), _result("X", stored_content),
              _call("Y"), _result("Y", "anchor Y"),
              {"role": "user", "content": "later durable turn"}]
    with engine_context(tmp_path, **options) as first:
        first.ingest(copy.deepcopy(original))
        # A legacy writer persisted this copy; current ingestion need not create it.
        for row in legacy:
            first._store.append(first.current_session_id, row, conversation_id=first._conversation_id)
        before = stored_rows(first)
    with engine_context(tmp_path, **options) as engine:
        new_pairs = [_call("N1"), _result("N1", "new N1"),
                     _call("N2"), _result("N2", "new N2")]
        host = [_call("X"), _result("X", replayed_content), *legacy[2:4], *new_pairs]
        unchanged_host = copy.deepcopy(host)
        engine.ingest(host)
        rows = stored_rows(engine)
        added = rows[len(before):]
        print("middle-window rows:", [(r["role"], r.get("tool_call_id"), r["content"]) for r in rows])
        print("middle-window reconciliation:", engine._last_ingest_reconciliation)
        assert rows[:len(before)] == before
        assert host == unchanged_host
        expected = new_pairs if already_durable else host
        assert [(r["role"], r.get("tool_call_id"), r["content"]) for r in added] == [
            (r["role"], r.get("tool_call_id"), r["content"]) for r in expected
        ]
        assert _grouped(rows)
        engine.ingest(copy.deepcopy(host))
        assert stored_rows(engine) == rows


@pytest.mark.parametrize("content", [
    ["list output " * 30],
    {"output": "dict output " * 30},
    int("9" * 250),
], ids=["untyped-list", "plain-dict", "number"])
def test_nonstring_source_cannot_hide_a_fresh_placeholder_pair(make_engine, content):  # noqa: F811
    """Canonical storage loses the content kind: fail closed for every non-string shape.

    Two newer outputs make X eligible by position; only source-kind provenance can reject it.
    Seed the durable representation directly so this also covers stores from older writers.
    """
    first = plain_engine(make_engine, fresh_tail_count=64)
    view = [SYSTEM, {"role": "user", "content": "original turn"}, _call("X"), _result("X", content),
            _call("Y"), _result("Y", "Y" * 300), _call("Z"), _result("Z", "Z" * 300)]
    for row in view:
        first._store.append(first._session_id, row, conversation_id=first._conversation_id)
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    before = stored(engine)
    assert before[3]["content"] == json.dumps(content, ensure_ascii=False, sort_keys=True)
    host = [dict(row, content=CLEARED) if i == 3 else row for i, row in enumerate(view)]

    engine._ingest_messages(host)

    rows = stored(engine)
    assert rows[:len(before)] == before
    added = rows[len(before):]
    assert any(row["role"] == "tool" and row["content"] == CLEARED for row in added)
    assert any(row.get("tool_calls") == host[2]["tool_calls"] for row in added)
    assert _grouped(rows)


@pytest.mark.parametrize("newer_tools", [0, 1, 2], ids=["newest", "second-newest", "true-salvage"])
@pytest.mark.parametrize("system", [False, True], ids=["effective-and-raw", "system-anchor"])
@pytest.mark.parametrize("proof", ["store", "host-rewrite", "native-recovery"])
def test_full_replay_uses_complete_list_placeholder_positions(make_engine, newer_tools, system, proof):  # noqa: F811
    """A full-length match cannot borrow a newest-two identity, even without a tool anchor.

    With two newer outputs the same occurrence really can be salvaged. Checking a candidate
    prefix's newest two instead of the complete list would duplicate that genuine replay.
    """
    first = plain_engine(make_engine, fresh_tail_count=64)
    view = ([SYSTEM] if system else []) + [{"role": "user", "content": "turn"},
                                         _call("X"), _result("X", "X" * 300)]
    for i in range(newer_tools):
        view += [_call(f"later-{i}"), _result(f"later-{i}", f"later output {i}")]
    first._ingest_messages(view)
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    if proof == "native-recovery":
        engine._remember_native_recovery_replay_snapshot_digest(engine._native_recovery_replay_snapshot_digest(view))
    before = stored(engine)
    host = [dict(row, content=CLEARED) if row.get("tool_call_id") == "X" else row for row in view]
    if proof == "host-rewrite":
        host = [dict(row, content="turn ") if row.get("role") == "user" else row for row in host]
        host += [{"role": "user", "content": "next turn"}]

    engine._ingest_messages(host)

    rows = stored(engine)
    assert rows[:len(before)] == before
    if newer_tools == 2:
        assert [row["content"] for row in rows[len(before):]] == (["next turn"] if proof == "host-rewrite" else [])
    else:
        added = rows[len(before):]
        assert any(row["role"] == "tool" and row["content"] == CLEARED for row in added)
        assert any(row.get("tool_calls") == _call("X")["tool_calls"] for row in added)
    assert _grouped(rows)
