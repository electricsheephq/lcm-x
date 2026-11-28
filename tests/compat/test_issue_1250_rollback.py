"""v0.30.4 can reopen and publish after #1250 re-owns provisional rows."""

import json

import pytest

from tests.compat.helpers import run_old
from tests.invariants.test_issue_1250_publication_paths import C, G, S, bind, digests, rotate_and_bind, rows
from tests.invariants.test_issue_1250_publication_reown import (
    GATEWAY, LATER, SESSION, SETTINGS, _bind_gateway, _publish, _seed_compacted,
    open_engine,  # noqa: F401
)

pytestmark = pytest.mark.issue(1250)
PREVIOUS_GA = {"v0.30.4": "0ab1160716f1c5167d17885a92aaa1ed10b0919f"}

OLD_RESUME = """
inputs = json.loads(data)
engine = LCMEngine(config=LCMConfig(database_path=db, **inputs["settings"]))
try:
    engine.on_session_start(inputs["session"], platform="desktop", context_length=200_000)
    assert engine._conversation_id == inputs["conversation"]
    state = engine._lifecycle
    before = state.get_by_conversation(engine._conversation_id).current_frontier_store_id
    nodes_before = engine._dag.get_session_node_count(engine._session_id)
    view = inputs["view"]
    engine.ingest(view)
    engine.compress(view)
    result = dict(status=engine.last_compression_status,
                  frontier_before=before,
                  frontier_after=state.get_by_conversation(engine._conversation_id).current_frontier_store_id,
                  nodes_before=nodes_before, nodes_after=engine._dag.get_session_node_count(engine._session_id),
                  rows=[list(row) for row in engine._store.connection.execute(
                      "SELECT store_id, session_id, conversation_id, role, content FROM messages ORDER BY store_id"
                  )],
                  integrity=[r[0] for r in engine._store.connection.execute("PRAGMA integrity_check")])
finally:
    engine.shutdown()
print(json.dumps(result))
"""


@pytest.mark.parametrize("rotated", [False, True], ids=["same-session", "rotated"])
def test_previous_ga_reopens_and_publishes_after_reownership(open_engine, tmp_path, old_tree, host_root, rotated):  # noqa: F811
    engine = open_engine()
    if rotated:
        bind(engine, S)
        host, expected = rotate_and_bind(engine)
        output = list(engine.compress(host))
        assert engine.last_compression_status == "compacted"
        assert digests(rows(engine, (S, C))) == digests(expected)
        assert any(r["conversation_id"] == G for r in engine._store.get_session_messages(S))
        session, conversation = C, G
    else:
        host = _seed_compacted(engine)
        _bind_gateway(engine)
        engine.ingest(host + LATER)
        output = _publish(engine, host + LATER)
        session, conversation = SESSION, GATEWAY
    before = [list(row) for row in engine._store.connection.execute(
        "SELECT store_id, session_id, conversation_id, role, content FROM messages ORDER BY store_id"
    )]
    frontier = engine._lifecycle.get_by_conversation(conversation).current_frontier_store_id
    engine.shutdown()

    new_turns = [
        {"role": "assistant", "content": "Rollback reader's first reply."},
        {"role": "user", "content": "Rollback reader's next request."},
    ]
    result = run_old(
        old_tree("v0.30.4", PREVIOUS_GA), tmp_path, host_root, OLD_RESUME,
        data=json.dumps(dict(settings=SETTINGS, session=session, conversation=conversation, view=output + new_turns)),
    )
    assert result["integrity"] == ["ok"]
    assert result["status"] == "compacted" and result["nodes_after"] > result["nodes_before"]
    assert result["frontier_after"] > result["frontier_before"] == frontier
    assert result["rows"][:len(before)] == before
    assert [(sid, owner, role, content) for _, sid, owner, role, content in result["rows"][len(before):]] == [
        (session, conversation, row["role"], row["content"]) for row in new_turns
    ]
