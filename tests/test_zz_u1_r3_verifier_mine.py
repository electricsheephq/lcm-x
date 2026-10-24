# ruff: noqa: F811
"""U1 r3 independent verifier probes (written before reading the lane's added tests).

Rule: after ANY undo, a restored alias_match row is active only if every source it cites
still passes D6 now; otherwise it stays retired and the key reads unresolved. A later
global override_key must never cite a gate-failed source.
"""

import json

import pytest

from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401


def _slot(store, key, owner):
    return [(r["entity_id"], r["method"], json.loads(r["source_store_ids"]))
            for r in active(store) if (r["subject_key"], r["owner_entity_id"]) == (key, owner)]


def _last(store):
    return json.loads(store.events_after()[-1]["payload_json"])


def _setup(bridge, namespace):
    api, ops, store, _, _ = bridge
    o1, o2 = ops.create_entity("owner one"), ops.create_entity("owner two")
    if namespace == "user":
        t1, key = o1, "user:self"
    else:
        t1, key = ops.create_entity("sister one"), "person:sister"
        ops.add_alias(ops.create_entity("sister two"), "my sister", "relation_phrase", owner_entity_id=o2)
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    pin, pin2 = ops.create_entity("pinned"), ops.create_entity("pinned two")
    a = publish(bridge, key)
    return api, ops, store, o1, o2, t1, key, pin, pin2, a


GATE = {
    "unverified": lambda o1, o2, a: {a: (o1, False, True)}.get,
    "not_first_person": lambda o1, o2, a: {a: (o1, True, False)}.get,
    "moved": lambda o1, o2, a: {a: (o2, True, True)}.get,
    "dropped": lambda o1, o2, a: {}.get,
}


def _assert_unresolved_and_uncited(api, ops, store, key, o1, a, pin):
    assert _slot(store, key, o1) == [], _slot(store, key, o1)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    try:
        ops.override_key(key, pin, owner_entity_id=o1)
    except api.EntityOpRefused:
        return
    assert a not in _last(store)["override"]["scope_store_ids"], _last(store)["override"]
    assert all(a not in cited for _, _, cited in _slot(store, key, o1))


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("gate", sorted(GATE))
def test_u1_each_gate_override_undo_without_rebuild(bridge, namespace, gate):
    api, ops, store, o1, o2, t1, key, pin, pin2, a = _setup(bridge, namespace)
    ok = {a: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=ok)
    assert _slot(store, key, o1) == [(t1, "alias_match", [a])]  # positive control
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin
    rebuild(bridge, sender_lookup=GATE[gate](o1, o2, a))
    ops.undo(ev)
    _assert_unresolved_and_uncited(api, ops, store, key, o1, a, pin)


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("gate", sorted(GATE))
def test_u1_stacked_overrides_undo_both_newest_first(bridge, namespace, gate):
    """Two overrides on one slot; gate fails; undo newest then oldest. Neither undo may revive."""
    api, ops, store, o1, o2, t1, key, pin, pin2, a = _setup(bridge, namespace)
    ok = {a: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=ok)
    ops.override_key(key, pin, owner_entity_id=o1)
    e1 = store.events_after()[-1]["event_id"]
    ops.override_key(key, pin2, owner_entity_id=o1)
    e2 = store.events_after()[-1]["event_id"]
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin2
    rebuild(bridge, sender_lookup=GATE[gate](o1, o2, a))
    ops.undo(e2)
    assert _slot(store, key, o1) == [], _slot(store, key, o1)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    ops.undo(e1)
    _assert_unresolved_and_uncited(api, ops, store, key, o1, a, pin)


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("gate", sorted(GATE))
def test_u1_undo_newest_while_healthy_then_gate_fail_then_undo_oldest(bridge, namespace, gate):
    api, ops, store, o1, o2, t1, key, pin, pin2, a = _setup(bridge, namespace)
    ok = {a: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=ok)
    ops.override_key(key, pin, owner_entity_id=o1)
    e1 = store.events_after()[-1]["event_id"]
    ops.override_key(key, pin2, owner_entity_id=o1)
    e2 = store.events_after()[-1]["event_id"]
    ops.undo(e2)  # healthy: restoring pin's override row is correct
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin
    rebuild(bridge, sender_lookup=GATE[gate](o1, o2, a))
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    ops.undo(e1)
    _assert_unresolved_and_uncited(api, ops, store, key, o1, a, pin)


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_u1_healthy_undo_still_restores(bridge, namespace):
    """Fail-safe must not over-retire: healthy undo (no gate failure) restores the alias row."""
    api, ops, store, o1, o2, t1, key, pin, pin2, a = _setup(bridge, namespace)
    ok = {a: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=ok)
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    rebuild(bridge, sender_lookup=ok)
    ops.undo(ev)
    assert _slot(store, key, o1) == [(t1, "alias_match", [a])]
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == t1


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("gate", sorted(GATE))
def test_u1_merge_undo_between_override_and_its_undo(bridge, namespace, gate):
    """A merge touching the pinned entity is undone first; then the override is undone."""
    api, ops, store, o1, o2, t1, key, pin, pin2, a = _setup(bridge, namespace)
    ok = {a: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=ok)
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    ops.merge(pin2, pin)
    m = store.events_after()[-1]["event_id"]
    rebuild(bridge, sender_lookup=GATE[gate](o1, o2, a))
    with pytest.raises(api.EntityOpRefused):
        ops.undo(ev)  # merge depends on pin
    ops.undo(m)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    ops.undo(ev)
    _assert_unresolved_and_uncited(api, ops, store, key, o1, a, pin)


@pytest.mark.parametrize("gate", sorted(GATE))
def test_u1_split_undo_after_gate_failure_never_resolves(bridge, gate):
    """Split moves the owner's relation alias; gate fails; undo split. No stale resolution."""
    api, ops, store, o1, o2, t1, key, pin, pin2, a = _setup(bridge, "person")
    ok = {a: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=ok)
    rel = next(r["alias_id"] for r in store.connection.execute(
        "SELECT * FROM lcm_entity_aliases WHERE entity_id=? AND retired_at IS NULL", (t1,)))
    new = ops.split(t1, [rel], [], "sister one prime")
    s = store.events_after()[-1]["event_id"]
    rebuild(bridge, sender_lookup=ok)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == new
    rebuild(bridge, sender_lookup=GATE[gate](o1, o2, a))
    ops.undo(s)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    _assert_unresolved_and_uncited(api, ops, store, key, o1, a, pin)


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("gate", sorted(GATE))
def test_u1_partial_gate_failure_two_sources(bridge, namespace, gate):
    """Override cites {a, c}; only a fails; undo must not make a resolve (c may be unresolved)."""
    api, ops, store, o1, o2, t1, key, pin, pin2, a = _setup(bridge, namespace)
    c = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (o1, True, True), c: (o1, True, True)}.get)
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    assert sorted(_last(store)["override"]["scope_store_ids"]) == sorted([a, c])
    later = {**GATE[gate](o1, o2, a).__self__, c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=later)
    assert api.resolve_key(store, key, c, owner_entity_id=o1) == pin
    ops.undo(ev)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    assert api.resolve_key(store, key, c, owner_entity_id=o1) in (None, t1)
    assert all(a not in cited for _, _, cited in _slot(store, key, o1)), _slot(store, key, o1)
    try:
        ops.override_key(key, pin, owner_entity_id=o1)
    except api.EntityOpRefused:
        return
    assert a not in _last(store)["override"]["scope_store_ids"]


def test_u1_unowned_override_source_invalidated_without_rebuild(bridge):
    """No owner: the override's source message's assertion is superseded; undo must not revive."""
    api, ops, store, _, _ = bridge
    alex = ops.create_entity("Alex")
    ops.add_alias(alex, "alex")
    pin = ops.create_entity("pinned")
    a = publish(bridge, "person:alex")
    rebuild(bridge)
    ops.override_key("person:alex", pin)
    ev = store.events_after()[-1]["event_id"]
    # Invalidate a's assertion source (no rebuild).
    store.connection.execute("UPDATE lcm_assertion_sources SET invalidated_at='2026-10-24' WHERE source_store_id=?", (a,))
    store.connection.commit()
    ops.undo(ev)
    assert api.resolve_key(store, "person:alex", a) is None
    assert [r for r in active(store) if r["subject_key"] == "person:alex" and a in json.loads(r["source_store_ids"])] == []
