"""#2029: a rotation child re-sent from cursor 0 links its predecessors' re-stored rows.

A survival-fit rotation writes no commit proof for the child, so a cold re-send into the child
stores the predecessors' rows again (the store growth stays open under #2029). The cursor-0
linker now annotates those copies against the recorded rotation chain: each copy is stored and
linked ``possible_replay_of`` to its earlier row, never skipped.
"""

import copy
import json

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_650_survival_fit_keeps_summary import _engine, _hidden_backlog, _turn, host  # noqa: F401


def _rotate(engine, old, new, view):
    engine.on_session_end(old, view)
    engine.on_session_start(new, platform="telegram", context_length=24_000, conversation_id="conv",
                            boundary_reason="compression", old_session_id=old)


def _links(engine, session_id):
    return engine._store._conn.execute(
        "SELECT r.store_id, t.session_id FROM message_relations r JOIN messages m ON m.store_id=r.store_id "
        "JOIN messages t ON t.store_id=r.related_store_id "
        "WHERE r.kind='possible_replay_of' AND m.session_id=? ORDER BY r.store_id", (session_id,)).fetchall()


def _session_rows(engine, session_id):
    return engine._store._conn.execute(
        "SELECT store_id, role, content FROM messages WHERE session_id=? ORDER BY store_id", (session_id,)).fetchall()


def test_held_fit_rotation_resend_links_every_predecessor_copy(tmp_path, host, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    engine = _engine(tmp_path)
    view = _hidden_backlog(engine, stamped_replies=True)
    first = engine.compress(view, current_tokens=host(view))
    _rotate(engine, "S", "S2", view)
    engine.ingest(copy.deepcopy(first))
    grown = [*first, *[r for i in range(12) for r in _turn(f"G{i:02d}", 20_000.0 + 10 * i, stamped_reply=True)],
             {"role": "user", "content": "[CUR] api variant", "timestamp": 30_000.0}]
    engine.ingest(grown)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_hold_fit_only_applies", lambda tokens: True)
        fitted = engine.compress(grown, current_tokens=host(grown))
    _rotate(engine, "S2", "S3", grown)
    engine.ingest(copy.deepcopy(fitted))
    engine.shutdown()

    cold = _engine(tmp_path)
    cold.on_session_start("S3", platform="telegram", context_length=24_000, conversation_id="conv")
    try:
        assert cold._rotation_predecessor_session_ids("S3") == ["S2", "S"]
        cold.ingest(copy.deepcopy(fitted))
        assert cold._ingest_failure_count == 0, cold._last_ingest_error
        rows = _session_rows(cold, "S3")
        links = _links(cold, "S3")
        # The re-sent view is stored (the growth stays open under #2029): nothing is skipped.
        assert len(rows) == 73
        assert len(links) == 72 and {target for _sid, target in links} == {"S", "S2"}
        unlinked = [(sid, content) for sid, _role, content in rows if sid not in {s for s, _t in links}]
        # The one unlinked row is the re-emitted summary marker, which has no stored original.
        assert len(unlinked) == 1 and unlinked[0][1].startswith("[Recent Summary")
    finally:
        cold.shutdown()


def _plain_engine(tmp_path, session_id, conversation_id="conv"):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start(session_id, platform="telegram", context_length=200_000, conversation_id=conversation_id)
    return engine


def _chain(engine, child, parent):
    """Record a verified compression boundary, as the rotation writes it (engine.py, on_session_start)."""
    engine._store.write_metadata_json([engine._rotation_predecessor_metadata_key(child)], json.dumps(parent))


def _seed(tmp_path, session_id, rows, conversation_id="conv"):
    engine = _plain_engine(tmp_path, session_id, conversation_id)
    engine.ingest(copy.deepcopy(rows))
    engine.shutdown()


def _child_ingest(tmp_path, rows, *, parent="P", conversation_id="conv"):
    engine = _plain_engine(tmp_path, "C", conversation_id)
    if parent:
        _chain(engine, "C", parent)
    engine.ingest(copy.deepcopy(rows))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    return engine


HISTORY = [{"role": "user", "content": "plan the trip", "timestamp": 1.0},
           {"role": "assistant", "content": "ok"}]


def test_non_ancestor_sessions_are_never_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    _seed(tmp_path, "X", HISTORY)  # same rows, earlier store ids, but not on the chain
    _seed(tmp_path, "P", HISTORY)
    child = _child_ingest(tmp_path, HISTORY)
    try:
        assert [target for _sid, target in _links(child, "C")] == ["P", "P"]
    finally:
        child.shutdown()


def test_without_a_recorded_chain_a_fresh_session_links_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    _seed(tmp_path, "P", HISTORY)
    child = _child_ingest(tmp_path, HISTORY, parent=None)
    try:
        assert len(_session_rows(child, "C")) == 2 and _links(child, "C") == []
    finally:
        child.shutdown()


def test_a_chain_into_a_foreign_conversation_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    _seed(tmp_path, "P", HISTORY, conversation_id="other")
    child = _child_ingest(tmp_path, HISTORY, conversation_id="conv")
    try:
        assert len(_session_rows(child, "C")) == 2 and _links(child, "C") == []
    finally:
        child.shutdown()


def test_unstamped_new_rows_are_stored_and_link_at_most_once(tmp_path, monkeypatch):
    """Accepted tradeoff, named in #2029's PR: an unstamped genuinely new row equal to an ancestor's row can be
    annotated as its replay (the same ambiguity the session-local linker already has). It is still stored, links
    at most one target, and a stamped new row with a different stamp is never linked."""
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    _seed(tmp_path, "P", HISTORY)
    new_turn = [{"role": "user", "content": "plan the trip", "timestamp": 50.0},
                {"role": "assistant", "content": "ok"}, {"role": "assistant", "content": "ok"}]
    child = _child_ingest(tmp_path, new_turn)
    try:
        rows = _session_rows(child, "C")
        links = _links(child, "C")
        assert [content for _sid, _role, content in rows] == ["plan the trip", "ok", "ok"]
        stamped_user = rows[0][0]
        assert stamped_user not in {sid for sid, _t in links}  # stamp veto: 50.0 is not 1.0
        assert len(links) == 1 and links[0][1] == "P"  # one ancestor "ok", one link; the second copy stays unlinked
    finally:
        child.shutdown()
