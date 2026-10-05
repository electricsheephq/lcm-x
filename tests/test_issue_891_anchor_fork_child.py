"""#891: the identity anchor must stop at the same fork boundary as host-uid."""

import sqlite3

import pytest

from tests.test_issue_436_identity_anchor import _engine, _rows, _state_db, _turns


def _fork_state_db(tmp_path, sessions):
    """Extend the #436 minimal fixture with optional host fork metadata."""
    _state_db(tmp_path, [row[:3] for row in sessions])
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute("ALTER TABLE sessions ADD COLUMN source TEXT")
        conn.execute("ALTER TABLE sessions ADD COLUMN model_config TEXT")
        conn.executemany(
            "UPDATE sessions SET source = ?, model_config = ? WHERE id = ?",
            [(source, config, sid) for sid, _parent, _end, source, config in sessions],
        )


@pytest.mark.parametrize("continuation", [False, True], ids=["branch", "compression-after-branch"])
def test_t1_branch_child_stops_chain_at_each_hop(tmp_path, continuation):
    sessions = [
        ("P", None, "compression", "telegram", None),
        ("B", "P", "compression", "telegram", '{"_branched_from": "P"}'),
    ]
    if continuation:
        sessions.append(("C", "B", None, "telegram", None))
    _fork_state_db(tmp_path, sessions)
    engine = _engine(tmp_path, "C" if continuation else "B")
    try:
        assert engine._identity_anchor_chain() == (["B"] if continuation else [])
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "source, config",
    [("tool", None), ("telegram", '{"_reset_from": "P"}'), ("telegram", '{"_delegate_from": "P"}')],
    ids=["tool", "reset", "delegate"],
)
def test_t2_other_fork_children_do_not_inherit(tmp_path, source, config):
    _fork_state_db(tmp_path, [
        ("P", None, "compression", "telegram", None),
        ("B", "P", None, source, config),
    ])
    engine = _engine(tmp_path, "B")
    try:
        assert engine._identity_anchor_chain() == []
    finally:
        engine.shutdown()


def test_t3_inherited_marker_naming_another_session_keeps_compression_parent(tmp_path):
    _fork_state_db(tmp_path, [
        ("P", None, "compression", "telegram", None),
        ("C", "P", None, "telegram", '{"_branched_from": "X"}'),
    ])
    engine = _engine(tmp_path, "C")
    try:
        assert engine._identity_anchor_chain() == ["P"]
    finally:
        engine.shutdown()


def test_t4_minimal_schema_keeps_compression_parent(tmp_path):
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    engine = _engine(tmp_path, "C")
    try:
        assert engine._identity_anchor_chain() == ["P"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("parent_end", ["compression", "user_exit"], ids=["branch-child", "sibling-control"])
def test_t5_copied_rows_belong_to_branch_without_parent_carry(tmp_path, parent_end):
    _fork_state_db(tmp_path, [
        ("P", None, parent_end, "telegram", None),
        ("B", "P", None, "telegram", '{"_branched_from": "P"}' if parent_end == "compression" else None),
    ])
    history = _turns(1, 4, 0.0)
    engine = _engine(tmp_path, "P")
    try:
        engine.ingest(history)
        parent_rows = _rows(engine, "P")
        assert len(parent_rows) == 8
        parent_ids = {int(row["store_id"]) for row in parent_rows}
        engine.on_session_start("B", platform="telegram", context_length=200_000, conversation_id="conv")
        live = [*[dict(row) for row in history], *_turns(10, 1, 600.0)]
        engine.ingest(live)
        branch_rows = _rows(engine, "B")
        assert len(branch_rows) == 10
        assert [row["content"] for row in branch_rows] == [row["content"] for row in live]
        assert not parent_ids & {int(row["store_id"]) for row in branch_rows}
        assert engine._identity_anchor_chain() == []
        assert engine._load_compression_carry_ranges() == []
    finally:
        engine.shutdown()
