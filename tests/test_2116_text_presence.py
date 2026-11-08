"""#2116 (v0.30.x, under #2467): a replaced reply below the cursor in a session longer than the 64-row audited tail.

The presence check stores a replaced assistant reply only when its replay identity (exact or #1286 cleanup form) is
stored nowhere in the scope and shown once in the view. Repeated text, an externalized or redacted stored reply in the
scope, and text past the text budget keep main's result (a positional replay): no loss and no copy against main.
Synthetic strings only."""
from __future__ import annotations

import logging
from collections import Counter

import pytest

from hermes_lcm import identity_anchor

from tests.test_anchor_unstored_before_cursor import PAD, _count, _engine, _key, _stored, _u

REPEATED = pytest.mark.xfail(strict=True, raises=AssertionError,
                             reason="#2467: a replaced reply whose text is stored elsewhere in the scope keeps "
                                    "main's result (one id is one invocation; text has no id)")
MARKED = pytest.mark.xfail(strict=True, raises=AssertionError,
                           reason="#2467: a scope holding an externalized or redacted stored reply keeps main's result "
                                  "(its identity would load a payload file or cannot prove absence)")


@pytest.fixture(autouse=True)
def _fresh_presence_notices(monkeypatch):
    monkeypatch.setattr(identity_anchor, "_PRESENCE_SHAPE_LOGGED", set())
    monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_LOGGED", set())
    monkeypatch.setattr(identity_anchor, "_PRESENCE_TEXT_LOGGED", set())


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


@REPEATED
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
    """The view also shows the new text past the cursor: the later row is stored by the normal path, so the replaced
    one keeps main's result and the text is stored once."""
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        new = _reply("a reply shown twice")
        _resend(engine, view, new, later=[_u("the next user turn", 3001), dict(new)])
        assert _count(engine, new["content"]) == 1
    finally:
        engine.shutdown()


@MARKED
@pytest.mark.parametrize("marker", [
    "[Externalized payload: 9000 chars; ref=lcm_000000000000_x.json]",
    "[Externalized LCM ingest payload: 9000 chars; ref=lcm_000000000000_y.json]",
    "[LCM sensitive redaction: name=password_assignment; len=8]",
], ids=["externalized", "ingest-placeholder", "lossy-redaction"])
def test_marked_stored_reply_named_limit_the_new_reply_is_stored(tmp_path, marker):
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        _mark_stored_reply(engine, 5, "an earlier reply " + marker + PAD)
        new = _reply("a brand new reply")
        _resend(engine, view, new)
        assert _count(engine, new["content"]) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("marker", [
    "[Externalized payload: 9000 chars; ref=lcm_000000000000_x.json]",
    "[LCM sensitive redaction: name=password_assignment; len=8]",
], ids=["externalized", "lossy-redaction"])
def test_a_marked_stored_reply_trips_to_mains_plan_and_loads_nothing(tmp_path, monkeypatch, marker):
    """A trip, not a guess: every reply keeps main's result (here nothing is new) before any stored reply's identity is
    computed, and nothing is stored twice. The tool audit runs as on main (tests/test_2116_r1_findings.py)."""
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        _mark_stored_reply(engine, 5, "an earlier reply " + marker + PAD)
        stored_identities = []
        real = type(engine)._message_replay_identity

        def spy(self, msg, *args, **kwargs):
            if kwargs.get("stored_row") and msg.get("role") == "assistant" and "store_id" in msg:
                stored_identities.append(msg["store_id"])
            return real(self, msg, *args, **kwargs)

        results = []
        unaccounted = type(engine)._identity_anchor_unaccounted

        def record(self, *args, **kwargs):
            stored_identities.clear()
            out = unaccounted(self, *args, **kwargs)
            results.append((out, list(stored_identities)))
            return out

        monkeypatch.setattr(type(engine), "_message_replay_identity", spy)
        monkeypatch.setattr(type(engine), "_identity_anchor_unaccounted", record)
        new = _reply("a brand new reply")
        _, after = _resend(engine, view, new)
        if not results:
            pytest.fail("setup: the presence check never ran")
        assert all(out == [] and not loaded for out, loaded in results), results
        assert max(after.values()) == 1
    finally:
        engine.shutdown()


def test_text_past_the_text_budget_trips_to_mains_plan(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    try:
        view = _warm(engine)
        monkeypatch.setattr(identity_anchor, "_PRESENCE_TEXT_BYTES", 2000, raising=False)
        new = _reply("a brand new reply")
        with caplog.at_level(logging.INFO, logger=identity_anchor.__name__):
            _resend(engine, view, new)
        assert _count(engine, new["content"]) == 0  # main's result: a positional replay
        assert any("text" in r.getMessage() and "presence window incomplete" in r.getMessage()
                   for r in caplog.records if r.name == identity_anchor.__name__)
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
