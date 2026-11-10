"""#2491, a named surplus limit (v0.30.0): a steer inserted before the carried parent tail re-stores the child's
unstamped reply.

A cleanup-only list is rotated into a compression child whose chain verifies (the host's state.db records the
parent). The child stores its own turn T6 with an unstamped reply. The host then inserts an unstamped row before the
carried parent tail, and the next warm ingest rewinds the cursor and stores T6's reply a second time with no relation.
Nothing is lost; the copy is silent. The strict xfail turns green with the fix (v0.30.x) and must then become an
ordinary test. The control is the same turn without the steer.
"""
import sqlite3

import pytest

from tests.test_issue_1286_cleanup_rewrite_proof import (  # noqa: F401 (fixtures)
    CHILD, CID, SID, STEER, _carried, _no_summaries, _pin_lcm_char_counter, _turns, host)

pytestmark = pytest.mark.issue(2491)


def _verified_chain(host):  # noqa: F811
    path = host.engine._state_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        conn.executemany("INSERT INTO sessions VALUES (?,?,?)", [(SID, None, "compression"), (CHILD, SID, None)])


def _child_turn_then_next_turn(host, steer):  # noqa: F811
    """The rotation into a verified child, its own turn T6, optionally the steer, then the next warm turn. Returns
    T6's reply content, its store ids in the child, and the relations that name them."""
    _verified_chain(host)
    try:  # the imported helpers assert; their failure must not read as the expected xfail
        _turns(host, "reasoning")
        result, before = host.cleanup_pass(preflight=True)
    except AssertionError as error:
        pytest.fail(f"setup: {error}")
    host.engine.on_session_end(SID, before)
    host.live = result
    host.engine.on_session_start(CHILD, boundary_reason="compression", old_session_id=SID, platform="telegram",
                                 conversation_id=CID, context_length=200_000)
    host.turn()  # T6, stored in the child with an unstamped reply
    if not host.engine._identity_anchor_chain():
        pytest.fail("setup: the child's chain must verify")
    if not _carried(host.engine._durable_commit_proof_payload(), SID):
        pytest.fail("setup: the rotation must carry the parent rows")
    reply = host.live[-1]["content"]
    if steer:
        host.live.insert(next(i for i, m in enumerate(host.live) if str(m.get("content")).startswith("[T4]")),
                         dict(STEER))
    host.turn()  # T7: the next warm ingest
    conn = host.engine._store._conn
    copies = [store_id for (store_id,) in conn.execute(
        "SELECT store_id FROM messages WHERE session_id = ? AND content = ? ORDER BY store_id", (CHILD, reply))]
    linked = {source for (source,) in conn.execute(
        "SELECT store_id FROM message_relations WHERE kind = 'possible_replay_of'")}
    next_user = conn.execute("SELECT count(*) FROM messages WHERE session_id = ? AND content = ?",
                             (CHILD, host.live[-2]["content"])).fetchone()[0]
    if not copies or next_user != 1:
        pytest.fail("setup: T6's reply and the next turn must each be stored")
    return copies, linked


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2491 named surplus limit (v0.30.0): a steer inserted before the carried parent tail "
                          "makes the next warm ingest store the child's unstamped reply again with no relation")
def test_a_steer_before_the_carried_tail_does_not_copy_the_childs_reply(host):  # noqa: F811
    copies, linked = _child_turn_then_next_turn(host, steer=True)
    if len(copies) > 2:
        pytest.fail(f"not the #2491 shape: {len(copies)} copies of the reply")
    assert len(copies) == 1 or copies[1] in linked, "#2491 known shape: T6's reply is stored twice, unlinked"


def test_the_next_turn_without_a_steer_stores_the_childs_reply_once(host):  # noqa: F811
    """Control (passes on main)."""
    copies, _linked = _child_turn_then_next_turn(host, steer=False)
    assert len(copies) == 1
