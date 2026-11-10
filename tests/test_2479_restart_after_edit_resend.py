"""#2479: cold turn-local reply consumption after edit-resend; synthetic fixtures only."""
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


@pytest.mark.parametrize("turns", [3, 40])
@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "unstamped"])
@pytest.mark.parametrize("resend", [True, False], ids=["resend", "plain"])
def test_restart(engines, turns, stamped, resend):
    engine = engines()
    view = _warm(engine, turns, stamped, resend)
    before = _counts(engine)
    engine = engines()
    engine.ingest(view)
    expected = Counter({_key(view[-3]): 1}) if resend and not stamped else Counter()
    assert _counts(engine) - before == expected
    assert not before - _counts(engine)
    before = _counts(engine)
    extra = [_u("after restart", 400), _reply("after reply", 401 if stamped else None)]
    view += extra
    engine.ingest(view)
    after = _counts(engine)
    assert after - before == Counter(map(_key, extra))
    assert not before - after and set(after.values()) == {1}
    assert not Counter(map(_key, view)) - after
    engine = engines()
    engine.ingest(view)
    assert _counts(engine) == after
    if resend and not stamped:
        duplicate = _reply("duplicate")
        view += [_u("copies", 500), duplicate, dict(duplicate), _u("replace again", 510), _reply("old again")]
        engine.ingest(view)
        view[-2:] = [_u("resent again", 520), _reply("new again")]
        view += [_u("later again", 530), _reply("later again reply")]
        engine.ingest(view)
        engine = engines()
        view.insert(len(view) - 4, dict(duplicate))
        engine.ingest(view)
        assert _counts(engine)[_key(duplicate)] == 3  # Two stored copies fund only two of the three replies.


@pytest.mark.parametrize("third", [False, True], ids=["two-stored", "third-new"])
def test_identical_replies_consume_distinct_copies(engines, third):
    engine = engines()
    view = [_u("prompt", 1), _reply("same"), _reply("same")]
    engine.ingest(view)
    assert _counts(engine)[_key(view[-1])] == 2
    engine = engines()
    engine.ingest(view + ([_reply("same")] if third else []))
    assert _counts(engine) == Counter(map(_key, view + ([_reply("same")] if third else [])))


def test_rewritten_reply_is_new(engines):
    engine = engines()
    view = [_u("prompt", 1), _reply("old")]
    engine.ingest(view)
    engine = engines()
    engine.ingest([view[0], _reply("rewritten")])
    assert _counts(engine) == Counter(map(_key, [*view, _reply("rewritten")]))


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


@pytest.fixture(params=[False, True], ids=["pairing", "pairing-disabled"])
def pairing(request, monkeypatch):
    if request.param:
        monkeypatch.setattr("hermes_lcm.ingest_ledger.cold_turn_pairs", lambda *a, **k: {})


def _new_reply(engine, view, reply):
    before = _counts(engine)
    def count():
        return sum(_key(r) == _key(reply) for r in _stored(engine)
                   if str(r.get("conversation_id") or "").strip() in ("", engine._conversation_id))
    expected = count() + 1
    engine.ingest(view)
    assert not before - _counts(engine)
    assert count() == expected


def test_p1_uid_conflict_keeps_new_reply(engines, pairing):
    engine, view = engines(), []
    for i in range(3):
        view += [_u(f"user {i}", 100 + i * 2), dict(_reply(f"reply {i}"), message_uid=f"reply-{i}")]
        engine.ingest(view)
    view[-2:] = [_u("resent", 300), _reply("replaced"), _u("next", 302), _reply("next reply")]
    engine.ingest(view)
    view[3] = dict(view[3], message_uid="new-reply")
    _new_reply(engines(), view, view[3])


def test_p1_conversation_keeps_new_reply(engines, pairing):
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


def test_p1_boundary_keeps_new_reply(engines, pairing, monkeypatch):
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


def test_p1_tool_keeps_new_reply(engines, pairing):
    engine = engines()
    view = [_u("prompt", 100), _call("C1", 101), _result("C1", "old result", 102), _reply("done")]
    engine.ingest(view)
    view[1:3] = [_call("C2", 201), _result("C2", "new result", 202)]
    _new_reply(engines(), view, view[-1])


def test_p1_partial_keeps_new_reply(engines, pairing):
    engine = engines()
    view = [_u("P", 100), _reply("reply"), _u("U", 200), _reply("yes")]
    engine.ingest(view)
    view[2] = dict(view[2], content=view[2]["content"] + "\n\nnew question")
    _new_reply(engines(), view, view[-1])
