"""#2116 (v0.30.x, under #2467): an UNSTAMPED reply the host replaced below the cursor, in a session longer than the
64-row audited tail, is not stored while the session runs; a restart stores it, with copies of the rest of the unstamped
history (#2479, #1674's class). Main's behaviour pinned: the ideal as strict xfails, and the guards and the stamped
case (stored warm) as passing tests. Synthetic strings only."""
from __future__ import annotations

from collections import Counter

import pytest

from hermes_lcm import identity_anchor

from tests.test_anchor_unstored_before_cursor import PAD, _count, _engine, _key, _stored, _u

UNSTAMPED = pytest.mark.xfail(strict=True, raises=AssertionError,
                              reason="#2116 (v0.30.x): an unstamped replaced reply past the audited tail is not stored "
                                     "while the session runs (main); revisit with #2479")


@pytest.fixture(autouse=True)
def _fresh_presence_notices(monkeypatch):
    monkeypatch.setattr(identity_anchor, "_PRESENCE_SHAPE_LOGGED", set())
    monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_LOGGED", set())
    monkeypatch.setattr(identity_anchor, "_PRESENCE_TEXT_LOGGED", set(), raising=False)


def _reply(text):
    return {"role": "assistant", "content": text + PAD}


def _warm(engine, turns=40):
    """``turns`` warm turns of a stamped user and an unstamped reply: more stored rows than the audited tail."""
    view = []
    for index in range(turns):
        view += [_u(f"T{index} user turn", 1000 + index), _reply(f"reply to T{index}")]
        engine.ingest(view)
    if len(_stored(engine)) <= 64:
        pytest.fail("setup: the session must hold more stored rows than the audited tail")
    return view


def _resend(engine, view, new_reply, *, later=()):
    """The host replaces the last turn with a resent user and ``new_reply``; one more turn (or ``later``) follows."""
    view[-2:] = [_u("the resent user turn", 3000), new_reply]
    view += list(later) or [_u("the next user turn", 3001), _reply("next")]
    before = Counter(_key(row) for row in _stored(engine))
    engine.ingest(view)
    after = Counter(_key(row) for row in _stored(engine))
    if before - after:
        pytest.fail(f"a stored row was lost: {before - after}")
    engine.ingest(view)
    if Counter(_key(row) for row in _stored(engine)) != after:
        pytest.fail("a repeat ingest changed the store")
    return before, after


def _mark_stored_reply(engine, index, text):
    """Rewrite one stored warm reply in place, as an externalizing or redacting writer leaves it."""
    conn = engine._store._conn
    rows = [row for row in _stored(engine) if row.get("role") == "assistant"]
    conn.execute("UPDATE messages SET content=? WHERE store_id=?", (text, int(rows[index]["store_id"])))
    conn.commit()


@UNSTAMPED
def test_a_replaced_reply_with_new_text_is_stored_once(tmp_path):
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        new = _reply("a brand new reply")
        _resend(engine, view, new)
        assert _count(engine, new["content"]) == 1
        assert _count(engine, "the resent user turn" + PAD) == 1
    finally:
        engine.shutdown()


def test_a_replaced_reply_whose_text_is_stored_outside_the_tail_is_not_stored_again(tmp_path):
    """The guard: an equal stored reply outside the tail accounts for the replaced one (as on main)."""
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        old = _reply("reply to T3")
        _resend(engine, view, dict(old))
        assert _count(engine, old["content"]) == 1
    finally:
        engine.shutdown()


@UNSTAMPED
def test_repeated_text_named_limit_a_new_occurrence_of_stored_text_is_stored(tmp_path):
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        old = _reply("reply to T3")
        _resend(engine, view, dict(old))
        assert _count(engine, old["content"]) == 2
    finally:
        engine.shutdown()


def test_a_replaced_reply_the_view_shows_again_at_the_cursor_is_stored_once(tmp_path):
    """The view also shows the new text past the cursor: the later row is stored by the normal path, and the text is
    stored once."""
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        new = _reply("a reply shown twice")
        _resend(engine, view, new, later=[_u("the next user turn", 3001), dict(new)])
        assert _count(engine, new["content"]) == 1
    finally:
        engine.shutdown()


def test_a_session_without_a_replaced_reply_is_unchanged(tmp_path):
    """The resent user alone: nothing else is stored (the reply is the same stored reply)."""
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        before, after = _resend(engine, view, dict(view[-1]))
        assert after - before == Counter({_key(_u("the resent user turn", 3000)): 1,
                                          _key(_u("the next user turn", 3001)): 1, _key(_reply("next")): 1})
    finally:
        engine.shutdown()


@pytest.mark.parametrize("turns", [3, 40])
def test_a_stamped_replaced_reply_is_stored_warm(tmp_path, turns):
    """The guard: a reply that reaches LCM-X with its host stamp is stored while the session runs (main's stamped
    identity-anchor path), and a repeat ingest changes nothing."""
    engine = _engine(tmp_path)
    try:
        view = []
        for index in range(turns):
            view += [_u(f"T{index} user turn", 1000 + index), dict(_reply(f"reply to T{index}"), timestamp=5000.0 + index)]
            engine.ingest(view)
        new = dict(_reply("a brand new reply"), timestamp=6000.0)
        _resend(engine, view, new, later=[_u("the next user turn", 3001), dict(_reply("next"), timestamp=6001.0)])
        assert _count(engine, new["content"]) == 1
    finally:
        engine.shutdown()
