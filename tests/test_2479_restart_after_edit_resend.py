"""#2479: a cold restart after an edit-resend re-stores the unstamped replies; fix C links those copies.

Fix C keeps #2305's ``possible_replay_of`` linking on when the durable ledger moves the cursor up from 0, so every copy
the cold restart stores is visible, not silent, and nothing is lost. The guards below pin that a genuinely new reply is
always stored: five shapes from review round 2 and three from the cross-model review of the withdrawn turn pairing.
They pass on main and stay as guards for any later attempt to pair replies. Synthetic fixtures only."""
from collections import Counter

import pytest

from tests.test_anchor_unstored_before_cursor import _a, _call, _engine, _key, _result, _stored, _u


@pytest.fixture
def engines(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    opened = []
    def restart():
        if opened:
            opened[-1].shutdown()
        opened.append(_engine(tmp_path))
        return opened[-1]
    yield restart
    opened[-1].shutdown()


def _reply(text, stamp=None):
    row = _a(text, stamp or 1)
    if stamp is None:
        row.pop("timestamp")
    return row


def _warm(engine, turns, stamped=False, resend=False, users=True):
    view = []
    for i in range(turns):
        user = _u(f"user {i}", 100 + i * 2)
        if not users:
            user.pop("timestamp")
        view += [user, _reply(f"reply {i}", 101 + i * 2 if stamped else None)]
        engine.ingest(view)
    if resend:
        view[-2:] = [_u("resent", 300), _reply("replaced", 301 if stamped else None)]
        if not users:
            view[-2].pop("timestamp")
        view += [_u("next", 302), _reply("next reply", 303 if stamped else None)]
        if not users:
            view[-2].pop("timestamp")
        engine.ingest(view)
    return view


def _counts(engine):
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    return Counter(_key(row) for row in _stored(engine))


def _visible(engine, view, before):
    """No host row is lost, and every row this ingest added beyond the host view's new occurrences is a linked copy
    (``before``: the stored counts before the ingest). Rows an edit-resend left in the store are not copies."""
    stored = _counts(engine)
    assert not Counter(map(_key, view)) - stored, "a host row is missing"
    assert not before - stored, "a stored row is missing"
    linked = {int(sid) for sid, in engine._store._conn.execute(
        "SELECT store_id FROM message_relations WHERE kind = 'possible_replay_of'")}
    added = stored - before
    for key, count in added.items():
        copies = count - max(0, Counter(map(_key, view))[key] - before[key])
        rows = [row for row in _stored(engine) if _key(row) == key][-count:]
        assert sum(int(row["store_id"]) in linked for row in rows) >= copies, (key[:2], copies)
    return stored


@pytest.mark.parametrize("turns", [3, 40])
@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "unstamped"])
@pytest.mark.parametrize("resend", [True, False], ids=["resend", "plain"])
def test_restart(engines, turns, stamped, resend):
    engine = engines()
    view = _warm(engine, turns, stamped, resend)
    before = _counts(engine)
    engine = engines()
    engine.ingest(view)
    restarted = _visible(engine, view, before)
    if stamped or not resend:
        assert restarted == before  # Stamped replies or no resend: the restart stores nothing new, as on main.
    before = restarted
    extra = [_u("after restart", 400), _reply("after reply", 401 if stamped else None)]
    view += extra
    engine.ingest(view)
    after = _visible(engine, view, before)
    assert after - before == Counter(map(_key, extra))
    engine = engines()
    engine.ingest(view)
    _visible(engine, view, after)


@pytest.mark.parametrize("third", [False, True], ids=["two-stored", "third-new"])
def test_identical_replies_are_kept(engines, third):
    engine = engines()
    view = [_u("prompt", 1), _reply("same"), _reply("same")]
    engine.ingest(view)
    assert _counts(engine)[_key(view[-1])] == 2
    before = _counts(engine)
    engine = engines()
    view += [_reply("same")] if third else []
    engine.ingest(view)
    assert _visible(engine, view, before)[_key(view[-1])] >= len(view) - 1


def test_rewritten_reply_is_new(engines):
    engine = engines()
    view = [_u("prompt", 1), _reply("old")]
    engine.ingest(view)
    before = _counts(engine)
    engine = engines()
    engine.ingest([view[0], _reply("rewritten")])
    assert _visible(engine, [view[0], _reply("rewritten")], before)[_key(_reply("rewritten"))] == 1


def test_ledger_off_keeps_main(engines, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    engine = engines()
    view = _warm(engine, 3, resend=True)
    before = _counts(engine)
    engine = engines()
    engine.ingest(view)
    assert sum((_counts(engine) - before).values()) == 4


def test_unstamped_users_keep_main(engines):
    engine = engines()
    view = _warm(engine, 3, resend=True, users=False)
    before = _counts(engine)
    engine = engines()
    engine.ingest(view)
    assert sum((_counts(engine) - before).values()) == 8


def test_tool_segment_keeps_main(engines):
    engine = engines()
    view = [_u("prompt", 1), _call("c", 2), _result("c", "result", 3), _reply("done")]
    engine.ingest(view)
    before = _counts(engine)
    engine = engines()
    engine.ingest(view)
    assert _counts(engine) == before


def _new_reply(engine, view, reply):
    before = _counts(engine)
    def count():
        return sum(_key(r) == _key(reply) for r in _stored(engine)
                   if str(r.get("conversation_id") or "").strip() in ("", engine._conversation_id))
    expected = count() + 1
    engine.ingest(view)
    assert not before - _counts(engine)
    assert count() == expected


def test_p1_uid_conflict_keeps_new_reply(engines):
    engine, view = engines(), []
    for i in range(3):
        view += [_u(f"user {i}", 100 + i * 2), dict(_reply(f"reply {i}"), message_uid=f"reply-{i}")]
        engine.ingest(view)
    view[-2:] = [_u("resent", 300), _reply("replaced"), _u("next", 302), _reply("next reply")]
    engine.ingest(view)
    view[3] = dict(view[3], message_uid="new-reply")
    _new_reply(engines(), view, view[3])


def test_p1_conversation_keeps_new_reply(engines):
    engine = engines()
    session, conversation = engine._session_id, engine._conversation_id
    view = [_u("prompt", 100), _reply("old")]
    engine.ingest(view)
    engine.on_session_start(session, conversation_id="B", context_length=1_000_000)
    engine.ingest([_reply("foreign")])
    engine.on_session_start(session, conversation_id=conversation, context_length=1_000_000)
    view += [_u("resent", 300), _reply("replaced"), _u("next", 302), _reply("next reply")]
    engine.ingest(view)
    view.insert(2, _reply("foreign"))
    _new_reply(engines(), view, view[2])


def test_p1_boundary_keeps_new_reply(engines, monkeypatch):
    engine = engines()
    view = [_u("A", 100), _reply("A reply")]
    engine.ingest(view)
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    view += [{"role": "user", "content": "B"}]
    engine.ingest(view)
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    view += [_reply("B reply"), _u("old", 200), _reply("old reply")]
    engine.ingest(view)
    view[-2:] = [_u("resent", 300), _reply("replaced"), _u("next", 302), _reply("next reply")]
    engine.ingest(view)
    view = [*view[:2], _reply("B reply"), *view[-4:]]
    _new_reply(engines(), view, view[2])


def test_p1_tool_keeps_new_reply(engines):
    engine = engines()
    view = [_u("prompt", 100), _call("C1", 101), _result("C1", "old result", 102), _reply("done")]
    engine.ingest(view)
    view[1:3] = [_call("C2", 201), _result("C2", "new result", 202)]
    _new_reply(engines(), view, view[-1])


def test_p1_partial_keeps_new_reply(engines):
    engine = engines()
    view = [_u("P", 100), _reply("reply"), _u("U", 200), _reply("yes")]
    engine.ingest(view)
    view[2] = dict(view[2], content=view[2]["content"] + "\n\nnew question")
    _new_reply(engines(), view, view[-1])


def _astra_store(engine, middle):
    """Warm turns, then an edit-resend of the last turn: the stored shape the cross-model review used."""
    view = [_u("first", 100), _reply("first reply"), _u("second", 200), *middle, _u("old", 300), _reply("old reply")]
    engine.ingest(view)
    view[-2:] = [_u("edited", 400), _reply("edited reply"), _u("next", 500), _reply("next reply")]
    engine.ingest(view)
    return view


def test_new_reply_with_a_normalized_prefix_is_stored(engines):
    """A new reply whose text differs from a stored reply only by volatile scaffolding that the identity digest
    strips (a model-switch note) is a different message: it is stored."""
    view = _astra_store(engines(), [_reply("OK")])
    view[3] = _reply("[Note: model was just switched from A to B]\nOK")
    _new_reply(engines(), view, view[3])


def test_new_reply_after_matched_tool_work_is_stored(engines):
    """A same-text reply that now follows a matched tool call and result, while the earlier commentary before the
    call is gone, is a new reply: it is stored."""
    view = _astra_store(engines(), [_reply("OK"), _call("C1", 201), _result("C1", "result", 202)])
    view[3:6] = [_call("C1", 201), _result("C1", "result", 202), _reply("OK")]
    _new_reply(engines(), view, view[5])


def test_new_reply_beside_a_uid_bound_twin_is_stored(engines):
    """A uid-less new reply with the same text as a stored reply that carries a host uid is stored. (The review's
    shape also drifted the lineage between the uid bindings and the occurrences; that only mattered to the withdrawn
    pairing, which read them under different lineages.)"""
    engine = engines()
    view = [_u("P", 100), dict(_reply("OK"), message_uid="old")]
    engine.ingest(view)
    view += [_u("old", 300), _reply("old reply")]
    engine.ingest(view)
    view[-2:] = [_u("resent", 400), _reply("replaced")]
    engine.ingest(view)
    view[1] = _reply("OK")
    _new_reply(engines(), view, view[1])
