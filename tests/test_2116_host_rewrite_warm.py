"""#2116 review r1 (Claude F2): the host rewrites an unstamped stored reply in place (Hermes' repair passes: a call
row that loses its calls, two replies merged, or changed internal whitespace) in the same ingest as an edit-resend of
the last turn. Warm, each rewritten text keeps main 9548dee7's count, and the replaced reply in the resent turn is
still stored once. Synthetic strings only."""
from __future__ import annotations

import json
import os
from collections import Counter

import pytest

from hermes_lcm import identity_anchor
from tests.test_anchor_unstored_before_cursor import PAD, _engine, _key, _stored, _u


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("_PRESENCE_SHAPE_LOGGED", "_PRESENCE_WINDOW_LOGGED", "_PRESENCE_TEXT_LOGGED"):
        monkeypatch.setattr(identity_anchor, name, set(), raising=False)


def _reply(text, ts=None):
    row = {"role": "assistant", "content": text + PAD}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _callrow(cid, text, ts=None):
    row = {"role": "assistant", "content": text + PAD, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _res(cid):
    return {"role": "tool", "content": "result " + cid, "tool_call_id": cid, "tool_name": "read_file"}


def _count_text(engine, text):
    return sum(1 for row in _stored(engine) if str(row.get("content") or "").strip() == (text).strip())


def _warm(engine, special, turns=40, at=30):
    view = []
    for index in range(turns):
        view.append(_u(f"T{index} user turn", 1000 + index))
        if index == at:
            view += special()
        else:
            view.append(_reply(f"reply to T{index}"))
        engine.ingest(view)
    return view


def _resend(view):
    view[-2:] = [_u("the resent user turn", 3000), _reply("a brand new reply")]
    view += [_u("the next user turn", 3001), _reply("next")]


SHAPES = {}


def shape(fn):
    SHAPES[fn.__name__] = fn
    return fn


@shape
def prune_unanswered_calls(stamped=False):
    """Hermes _prune_unanswered_tool_calls / _sanitize_tool_pairs: a stored call row loses its calls in place (its result
    row is gone), content kept."""
    ts = 1030.5 if stamped else None
    def special():
        return [_callrow("c30", "I will check the file", ts), _res("c30"), _reply("reply to T30")]

    def mutate(view):
        i = next(k for k, m in enumerate(view) if m.get("tool_calls") and m["tool_calls"][0]["id"] == "c30")
        view[i].pop("tool_calls")
        del view[i + 1]
    return special, mutate, ["I will check the file" + PAD]


@shape
def merge_consecutive_assistants(stamped=False):
    """Hermes _merge_consecutive_assistants: two stored unstamped replies folded into the first, joined by \\n."""
    def special():
        return [_reply("interim reply"), _reply("final reply")]

    def mutate(view):
        i = next(k for k, m in enumerate(view) if m.get("content") == "interim reply" + PAD)
        view[i]["content"] = "\n".join(p for p in (view[i]["content"].strip(), view[i + 1]["content"].strip()) if p)
        del view[i + 1]
    return special, mutate, ["interim reply" + PAD, "final reply" + PAD,
                             ("interim reply" + PAD).strip() + "\n" + ("final reply" + PAD).strip()]


@shape
def internal_whitespace(stamped=False):
    def special():
        return [_reply("reply  to T30 spaced")]

    def mutate(view):
        i = next(k for k, m in enumerate(view) if m.get("content") == "reply  to T30 spaced" + PAD)
        view[i]["content"] = "reply to T30 spaced" + PAD
    return special, mutate, ["reply  to T30 spaced" + PAD, "reply to T30 spaced" + PAD]



def _resend_in_turn(view):
    """The edit-resend replaces the last turn's user and adds the new reply after that turn's (rewritten) rows."""
    last_user = max(k for k, m in enumerate(view) if m.get("role") == "user")
    view[last_user] = _u("the resent user turn", 3000)
    view += [_reply("a brand new reply"), _u("the next user turn", 3001), _reply("next")]


def _ingest_twice(engine, view):
    engine.ingest(view)
    after = Counter(_key(r) for r in _stored(engine))
    engine.ingest(view)
    assert Counter(_key(r) for r in _stored(engine)) == after, "a repeat ingest changed the store"
    return after


def _rewrite_then_resend(tmp_path, name, placement, restart):
    special, mutate, texts = SHAPES[name]()
    engine = _engine(tmp_path)
    try:
        view = _warm(engine, special, at=30 if placement == "another-turn" else 39)
        before = Counter(_key(r) for r in _stored(engine))
        mutate(view)
        (_resend if placement == "another-turn" else _resend_in_turn)(view)
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        after = _ingest_twice(engine, view)
        assert not before - after, "a stored row was lost"
        counts = [_count_text(engine, t) for t in [*texts, "a brand new reply" + PAD]]
        return counts, after - before
    finally:
        engine.shutdown()


# Main 9548dee7's counts: each rewritten text, then the replaced reply (this file passes on main, except that main
# stores no replaced reply warm, #2116, so the replaced reply's count is checked separately).
MAIN = {
    ('prune_unanswered_calls', 'another-turn', False): [1],
    ('prune_unanswered_calls', 'another-turn', True): [2],
    ('prune_unanswered_calls', 'missed-turn', False): [1],
    ('prune_unanswered_calls', 'missed-turn', True): [2],
    ('merge_consecutive_assistants', 'another-turn', False): [1, 1, 0],
    ('merge_consecutive_assistants', 'another-turn', True): [1, 1, 1],
    ('merge_consecutive_assistants', 'missed-turn', False): [1, 1, 0],
    ('merge_consecutive_assistants', 'missed-turn', True): [1, 1, 1],
    ('internal_whitespace', 'another-turn', False): [1, 0],
    ('internal_whitespace', 'another-turn', True): [1, 1],
    ('internal_whitespace', 'missed-turn', False): [1, 0],
    ('internal_whitespace', 'missed-turn', True): [1, 1],
}
MAIN_REPLY_ON_MAIN = {
    ('prune_unanswered_calls', 'another-turn', False): 0,
    ('prune_unanswered_calls', 'another-turn', True): 1,
    ('prune_unanswered_calls', 'missed-turn', False): 0,
    ('prune_unanswered_calls', 'missed-turn', True): 1,
    ('merge_consecutive_assistants', 'another-turn', False): 0,
    ('merge_consecutive_assistants', 'another-turn', True): 1,
    ('merge_consecutive_assistants', 'missed-turn', False): 0,
    ('merge_consecutive_assistants', 'missed-turn', True): 1,
    ('internal_whitespace', 'another-turn', False): 0,
    ('internal_whitespace', 'another-turn', True): 1,
    ('internal_whitespace', 'missed-turn', False): 1,
    ('internal_whitespace', 'missed-turn', True): 1,
}
ON_MAIN = not hasattr(identity_anchor, "_PRESENCE_TEXT_BYTES")


@pytest.mark.parametrize("restart", [False, True], ids=["warm", "restart"])
@pytest.mark.parametrize("placement", ["another-turn", "missed-turn"])
@pytest.mark.parametrize("name", ["prune_unanswered_calls", "merge_consecutive_assistants", "internal_whitespace"])
def test_a_host_rewrite_keeps_mains_counts_and_the_replaced_reply_is_stored_once(tmp_path, name, placement, restart):
    counts, added = _rewrite_then_resend(tmp_path, name, placement, restart)
    print(f"\nREWRITE {name} {placement} restart={restart} counts={counts} added={sum(added.values())}")
    key = (name, placement, restart)
    dump = os.environ.get("LCM_REWRITE_DUMP")
    if dump:  # evidence: the added rows as a multiset, compared across main and the PR
        with open(os.path.join(dump, "-".join(map(str, key)) + ".json"), "w") as f:
            json.dump(sorted([list(k), v] for k, v in added.items()), f)
    assert counts[:-1] == MAIN[key]
    assert counts[-1] == (MAIN_REPLY_ON_MAIN[key] if ON_MAIN else 1)  # the PR stores the replaced reply once
