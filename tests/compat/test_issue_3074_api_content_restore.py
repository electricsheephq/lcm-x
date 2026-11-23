"""#3074: Hermes keeps user sidecars when it re-stores the active transcript."""

from contextlib import closing
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.issue(3074),
    pytest.mark.skipif(find_spec("hermes_state") is None, reason="requires a real Hermes SessionDB"),
]

SESSION = "gate-3074"
LATE_USER = "User turn with late context"
PLAIN_USER = "User turn without context"
RESUMED_USER = "User turn after restart"
CONTEXT = "Synthetic café context\nSecond line."


def _user_sidecars(rows):
    return [
        (row["content"], row["api_content"].encode("utf-8") if row.get("api_content") is not None else None)
        for row in rows if row["role"] == "user"
    ]


def _commit_pass(db, engine, history, original_sidecars, *, phase, plain_user):
    from hermes_lcm.tokens import count_messages_tokens

    before = db.get_messages(SESSION)
    compacted = engine.compress(history, current_tokens=count_messages_tokens(history), force=True)
    assert engine.last_compression_status == "compacted", (phase, engine._last_compression_noop_reason)
    assert compacted != history, phase

    inserted = db.archive_and_compact(SESSION, compacted)
    active = db.get_messages(SESSION)
    assert inserted == len(active) == len(compacted), phase
    assert {row["id"] for row in before}.isdisjoint(row["id"] for row in active), phase

    sidecars = _user_sidecars(active)
    assert {LATE_USER, plain_user} <= {content for content, _ in sidecars}, phase
    for content, sidecar in sidecars:
        # A generated summary row has no original sidecar.
        expected = original_sidecars.get(content)
        assert sidecar == expected, (phase, content)


@pytest.mark.parametrize("uid_mode", ["off", "shadow"], ids=["uid-off", "uid-carry"])
def test_re_stored_user_sidecars_survive_compaction_and_restart(tmp_path, monkeypatch, host_root, uid_mode):
    # Hermes' tools package must resolve ahead of LCM-X's tools.py.
    monkeypatch.syspath_prepend(str(host_root))
    from agent.turn_context import _stamp_api_content_sidecar
    from hermes_state import SessionDB

    from hermes_lcm import engine as engine_module
    from hermes_lcm.config import host_message_uid_mode
    from hermes_lcm.host_uid_emit import host_uid_capable
    from tests.harness.engine import engine_context

    if uid_mode == "shadow" and not host_uid_capable():
        pytest.skip("this Hermes build has no host message UID support")
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    assert host_message_uid_mode() == uid_mode
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Synthetic leaf summary.", 1))

    def open_engine():
        return engine_context(
            tmp_path, window=200_000, session_id=SESSION, fresh_tail_count=6, leaf_chunk_tokens=120,
            embeddings_enabled=False, temporal_rollups_enabled=False,
        )

    db_path = tmp_path / "state.db"
    with closing(SessionDB(db_path=db_path)) as db, open_engine() as engine:
        db.create_session(SESSION, source="cli")
        for i in range(16):
            db.append_message(SESSION, "user", f"Question {i}:" + " alpha beta gamma delta" * 40)
            db.append_message(SESSION, "assistant", f"Answer {i}.")
        db.append_message(SESSION, "user", PLAIN_USER)
        db.append_message(SESSION, "assistant", "First tail reply.")
        db.append_message(SESSION, "user", LATE_USER)
        history = db.get_messages_as_conversation(SESSION, include_row_ids=True)
        current = history[-1]
        assert current["content"] == LATE_USER and "api_content" not in current

        # The turn-start preflight can ingest the user row before the host composes its API sidecar.
        engine.ingest(history)
        assert engine._last_active_replay_messages[-1]["content"] == LATE_USER
        assert "api_content" not in engine._last_active_replay_messages[-1]

        agent = SimpleNamespace(session_id=SESSION, _session_db=db)
        _stamp_api_content_sidecar(agent, history, len(history) - 1, "", CONTEXT, preflight_compressed=False)
        assert CONTEXT in current["api_content"]
        reply_id = db.append_message(SESSION, "assistant", "Reply after the sidecar stamp.")
        reply = db.get_messages_as_conversation(SESSION, include_row_ids=True)[-1]
        assert reply["_row_id"] == reply_id
        history.append(reply)

        original_sidecars = dict(_user_sidecars(history))
        assert list(original_sidecars.items()) == _user_sidecars(db.get_messages(SESSION))
        _commit_pass(db, engine, history, original_sidecars, phase="before restart", plain_user=PLAIN_USER)

    with closing(SessionDB(db_path=db_path)) as db, open_engine() as engine:
        history = db.get_messages_as_conversation(SESSION, include_row_ids=True)
        engine.ingest(history)
        # Advance the fresh tail so another raw exchange is eligible for compaction.
        db.append_message(SESSION, "user", RESUMED_USER)
        db.append_message(SESSION, "assistant", "Reply after restart.")
        history = db.get_messages_as_conversation(SESSION, include_row_ids=True)
        original_sidecars[RESUMED_USER] = None
        _commit_pass(db, engine, history, original_sidecars, phase="after restart", plain_user=RESUMED_USER)
