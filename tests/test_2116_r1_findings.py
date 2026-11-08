"""#2116 review r1: the reply check never adds a copy or loses a row against main 9548dee7.

Every test here also passes on main, where no reply is a candidate: each asserts main's answer. Synthetic strings only."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from hermes_lcm import identity_anchor

from tests.test_anchor_unstored_before_cursor import PAD, SESSION, _call, _engine, _key, _result, _u

OOB = "\n[OUT-OF-BAND USER MESSAGE]\nkeep this instruction\n[/OUT-OF-BAND USER MESSAGE]"
NEW = {"role": "assistant", "content": "a brand new reply" + PAD}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("_PRESENCE_SHAPE_LOGGED", "_PRESENCE_WINDOW_LOGGED", "_PRESENCE_TEXT_LOGGED"):
        monkeypatch.setattr(identity_anchor, name, set(), raising=False)


def _presence(engine, view, identities=None):
    identities = identities or [engine._message_replay_identity(m) for m in view]
    return engine._identity_anchor_unaccounted(
        view, view, len(view), 0, lambda i: identities[i], {}, True, set(), {"replayed": set()}, {}, (), [])


# (0) A reply whose identity as a stored row differs from its incoming identity cannot be proved absent from stored
# identities, so it keeps main's result. The assistant-text branches of _message_replay_identity where the two forms
# differ: an out-of-band block on a row with a tool_call_id (stripped on the stored side always, on the incoming side
# only once this session holds the row), a survival-fit projection (#582: the incoming row is its source), and a
# payload or redaction marker (restored or loaded on the stored side; such a reply is never normalized here). The host
# rewrite, the summary carrier, the pruned-tool alias and persisted-output recovery apply to user or tool rows only.

@pytest.mark.parametrize("where", ["predecessor", "nowhere"])
def test_r1_0_an_out_of_band_reply_with_a_call_id_is_not_copied(tmp_path, where):
    engine = _engine(tmp_path)
    try:
        engine._record_rotation_predecessor(SESSION, "P")
        row = {"role": "assistant", "content": "a stored reply" + PAD + OOB, "tool_call_id": "oob-occurrence"}
        if where == "predecessor":
            engine._store.append_batch("P", [_u("older user", 1), dict(row)])
        assert _presence(engine, [dict(row)]) == []
    finally:
        engine.shutdown()


def test_r1_0_an_out_of_band_reply_without_a_call_id_is_unchanged(tmp_path):
    """No tool_call_id: no strip on either side, so the stored copy accounts for it (as on main)."""
    engine = _engine(tmp_path)
    try:
        engine._record_rotation_predecessor(SESSION, "P")
        row = {"role": "assistant", "content": "a stored reply" + PAD + OOB}
        engine._store.append_batch("P", [_u("older user", 1), dict(row)])
        assert _presence(engine, [dict(row)]) == []
    finally:
        engine.shutdown()


def test_r1_0_a_survival_fit_projection_of_a_stored_reply_is_not_copied(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch("P", [_u("older user", 1), {"role": "assistant", "content": "the source" + PAD}])
        source = next(r for r in engine._store.get_session_messages("P") if r.get("role") == "assistant")
        projected = {"role": "assistant", "content": "the source, projected"}
        real = type(engine)._survival_projection_source
        monkeypatch.setattr(type(engine), "_survival_projection_source", lambda self, m, *a, **k: (
            source if m.get("content") == projected["content"] else real(self, m, *a, **k)))
        assert _presence(engine, [dict(projected)]) == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize("marker", ["[Externalized payload: 9000 chars; ref=lcm_000000000000_x.json]",
                                    "[Externalized LCM ingest payload: 9000 chars; ref=lcm_000000000000_y.json]",
                                    "[LCM sensitive redaction: name=password_assignment; len=8]"])
def test_r1_0_a_marked_reply_is_never_normalized_as_a_stored_row(tmp_path, monkeypatch, marker):
    engine = _engine(tmp_path)
    try:
        row = {"role": "assistant", "content": "a reply " + marker + PAD}
        identities = [engine._message_replay_identity(row)]
        seen = []
        real = type(engine)._message_replay_identity
        monkeypatch.setattr(type(engine), "_message_replay_identity", lambda self, m, *a, **k: (
            seen.append(k.get("stored_row")) or real(self, m, *a, **k)))
        assert _presence(engine, [row], identities) == []
        assert seen == []
    finally:
        engine.shutdown()


def test_r1_0_any_one_sided_normalization_keeps_mains_result(tmp_path, monkeypatch):
    """The class: whatever makes the stored-row form differ, the reply is not a candidate."""
    engine = _engine(tmp_path)
    try:
        real = type(engine)._message_replay_identity

        def one_sided(self, m, *a, **k):
            identity = real(self, m, *a, **k)
            return (identity[0], identity[1] + " (stored form)", *identity[2:]) if k.get("stored_row") else identity

        monkeypatch.setattr(type(engine), "_message_replay_identity", one_sided)
        assert _presence(engine, [dict(NEW)]) == []
    finally:
        engine.shutdown()


# (1) A text trip gives every reply main's result and leaves the tool audit as on main.

def _tool_shape(engine, shape):
    """#2466 tool shapes behind a new reply: the reply is row 0."""
    pair = [_call("call_x", 3), _result("call_x", "the result", 4)]
    if shape == "moved-pair":
        pass
    elif shape == "stored-pair":
        engine._store.append_batch(SESSION, [dict(r) for r in pair])
    elif shape == "predecessor-pair":
        engine._record_rotation_predecessor(SESSION, "P")
        engine._store.append_batch("P", [dict(r) for r in pair])
    elif shape == "stored-call-only":
        engine._store.append_batch(SESSION, [dict(pair[0])])
    elif shape == "reused-id":
        pair = [*pair, _call("call_x", 5), _result("call_x", "another result", 6)]
    elif shape == "two-pairs":
        pair = [*pair, _call("call_y", 5), _result("call_y", "the other result", 6)]
    return [dict(NEW), *pair]


SHAPES = ["moved-pair", "stored-pair", "predecessor-pair", "stored-call-only", "reused-id", "two-pairs"]
MAIN = {"moved-pair": [1, 2], "stored-pair": [], "predecessor-pair": [], "stored-call-only": [2],
        "reused-id": [], "two-pairs": [1, 2, 3, 4]}  # main 9548dee7's tool plans (this file passes there)


def _rows(view, plan):
    return Counter(_key(view[i]) for i in plan) if plan is not None else None


@pytest.mark.parametrize("trip", ["marked-stored-reply", "text-budget-zero", "read-error"])
@pytest.mark.parametrize("shape", SHAPES)
def test_r1_1_under_a_text_trip_the_tool_plan_is_mains(tmp_path, monkeypatch, shape, trip):
    engine = _engine(tmp_path)
    try:
        view = _tool_shape(engine, shape)
        if trip == "marked-stored-reply":
            engine._store.append_batch(SESSION, [_u("older user", 1), {
                "role": "assistant", "content": "an earlier reply [LCM sensitive redaction: name=x; len=8]" + PAD}])
        elif trip == "text-budget-zero":
            monkeypatch.setattr(identity_anchor, "_PRESENCE_TEXT_BYTES", 0, raising=False)
        else:
            def fail(self, *a, **k):
                raise OSError("synthetic reply read error")
            monkeypatch.setattr(type(engine), "_active_cleanup_replay_identity", fail, raising=False)
        assert _rows(view, _presence(engine, view)) == _rows(view, MAIN[shape])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", SHAPES)
def test_r1_1_without_a_trip_only_the_new_reply_is_added_to_mains_tool_plan(tmp_path, shape):
    engine = _engine(tmp_path)
    try:
        view = _tool_shape(engine, shape)
        plan = _presence(engine, view)
        tools = [i for i in plan if i != 0]
        assert _rows(view, tools) == _rows(view, MAIN[shape])
        assert 0 in plan or not hasattr(identity_anchor, "_PRESENCE_TEXT_BYTES")  # main: replies are not candidates
    finally:
        engine.shutdown()


# (2) Charge before compute.

def test_r1_2_an_over_budget_reply_costs_no_identity_or_cleanup_work(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        big = {"role": "assistant", "content": "y" * 4_000_001}
        identities = [engine._message_replay_identity(big)]
        calls = []
        real_identity = type(engine)._message_replay_identity
        real_cleanup = type(engine)._active_cleanup_replay_identity
        monkeypatch.setattr(type(engine), "_message_replay_identity",
                            lambda self, *a, **k: calls.append("identity") or real_identity(self, *a, **k))
        monkeypatch.setattr(type(engine), "_active_cleanup_replay_identity",
                            lambda self, *a, **k: calls.append("cleanup") or real_cleanup(self, *a, **k))
        assert _presence(engine, [big], identities) == []
        assert calls == []
    finally:
        engine.shutdown()
