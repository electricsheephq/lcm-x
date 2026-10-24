# ruff: noqa: F811  # Imported pytest fixture is intentionally injected by name.
"""N1 verifier probes (independent, written before reading the lane's ported tests).

Rule (a): with no active override, a key that rebuild retired for failing a D6 gate never
resolves and is never cited. Rule (c): override_key cites only the owner's CURRENT sources.
"""

import json

import pytest

from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401


def _keys(store):
    return {(r["subject_key"], r["owner_entity_id"]): (r["entity_id"], r["method"], json.loads(r["source_store_ids"]))
            for r in active(store)}


def _last_override(store):
    return json.loads(store.events_after()[-1]["payload_json"])["override"]


def _world(bridge, namespace):
    api, ops, store, _, _ = bridge
    o1, o2 = ops.create_entity("owner one"), ops.create_entity("owner two")
    if namespace == "user":
        t1, t2 = o1, o2
        key = "user:self"
    else:
        t1, t2 = ops.create_entity("sister one"), ops.create_entity("sister two")
        key = "person:sister"
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    ops.add_alias(t2, "my sister", "relation_phrase", owner_entity_id=o2)
    pin = ops.create_entity("pinned")
    a = publish(bridge, key, scope="P1")
    b = publish(bridge, key, scope="P2")
    return api, ops, store, o1, o2, t1, t2, pin, a, b, key


GATES = {
    "unverified": lambda o1, o2, a, b: {a: (o1, False, True), b: (o2, True, True)}.get,
    "not_first_person": lambda o1, o2, a, b: {a: (o1, True, False), b: (o2, True, True)}.get,
    "moved": lambda o1, o2, a, b: {a: (o2, True, True), b: (o2, True, True)}.get,
    "dropped": lambda o1, o2, a, b: {b: (o2, True, True)}.get,
}


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("gate", sorted(GATES))
def test_mine_gate_failure_never_resolves_or_cites(bridge, namespace, gate):
    api, ops, store, o1, o2, t1, t2, pin, a, b, key = _world(bridge, namespace)
    rebuild(bridge, sender_lookup={a: (o1, True, True), b: (o2, True, True)}.get)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == t1  # positive control
    later = GATES[gate](o1, o2, a, b)
    rebuild(bridge, sender_lookup=later)
    rebuild(bridge, sender_lookup=later)  # idempotent second pass
    assert (key, o1) not in _keys(store)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    if gate == "moved":
        assert api.resolve_key(store, key, a, owner_entity_id=o2) == t2
    # Never cited: o1's global override must not cite a.
    try:
        ops.override_key(key, pin if namespace != "user" else pin, owner_entity_id=o1)
    except api.EntityOpRefused:
        return
    assert a not in _last_override(store)["scope_store_ids"]
    rebuild(bridge, sender_lookup=later)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    assert a not in _keys(store).get((key, o1), (None, None, []))[2]


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("gate", sorted(GATES))
@pytest.mark.parametrize("rebuild_after_undo", [False, True])
def test_mine_override_before_retirement_then_undone(bridge, namespace, gate, rebuild_after_undo):
    api, ops, store, o1, o2, t1, t2, pin, a, b, key = _world(bridge, namespace)
    first = {a: (o1, True, True), b: (o2, True, True)}.get
    rebuild(bridge, sender_lookup=first)
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    assert _last_override(store)["scope_store_ids"] == [a]
    rebuild(bridge, sender_lookup=first)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin  # control
    later = GATES[gate](o1, o2, a, b)
    rebuild(bridge, sender_lookup=later)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None, "retired override still binds"
    ops.undo(ev)
    if rebuild_after_undo:
        rebuild(bridge, sender_lookup=later)
    got = api.resolve_key(store, key, a, owner_entity_id=o1)
    assert got is None, f"after undo, gate-failed key resolves to {got} (t1={t1}); active={_keys(store)}"
    try:
        ops.override_key(key, pin, owner_entity_id=o1)
    except api.EntityOpRefused:
        return
    assert a not in _last_override(store)["scope_store_ids"], _last_override(store)


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_mine_active_override_does_not_admit_gate_failed_sibling_source(bridge, namespace):
    """o1 keeps an active override citing c; a separate o1 source a fails a gate.
    The active override must not make a resolve through the retired baseline (a is not cited)."""
    api, ops, store, o1, o2, t1, t2, pin, a, b, key = _world(bridge, namespace)
    c = publish(bridge, key, scope="P1")
    first = {a: (o1, True, True), b: (o2, True, True), c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=first)
    ops.override_key(key, pin, scope_store_ids=None, owner_entity_id=o1)
    assert _last_override(store)["scope_store_ids"] == sorted([a, c])
    rebuild(bridge, sender_lookup=first)
    later = {a: (o1, False, True), b: (o2, True, True), c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=later)
    assert _keys(store)[(key, o1)] == (pin, "override", [c])
    assert api.resolve_key(store, key, c, owner_entity_id=o1) == pin
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    ops.override_key(key, t1 if namespace != "user" else pin, owner_entity_id=o1)
    assert a not in _last_override(store)["scope_store_ids"]


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_mine_two_owners_share_phrase_override_on_one_retire_other(bridge, namespace):
    """o1 holds an active override; o2's only source fails a gate. o1's override must not
    admit o2's retired baseline, and o2 never resolves or cites the retired source."""
    api, ops, store, o1, o2, t1, t2, pin, a, b, key = _world(bridge, namespace)
    first = {a: (o1, True, True), b: (o2, True, True)}.get
    rebuild(bridge, sender_lookup=first)
    ops.override_key(key, pin, owner_entity_id=o1)
    rebuild(bridge, sender_lookup=first)
    assert api.resolve_key(store, key, b, owner_entity_id=o2) == t2
    later = {a: (o1, True, True), b: (o2, False, True)}.get
    rebuild(bridge, sender_lookup=later)
    assert (key, o2) not in _keys(store)
    assert api.resolve_key(store, key, b, owner_entity_id=o2) is None
    assert api.resolve_key(store, key, b, owner_entity_id=o1) is None
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin
    try:
        ops.override_key(key, pin, owner_entity_id=o2)
    except api.EntityOpRefused:
        return
    assert b not in _last_override(store)["scope_store_ids"]


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_mine_both_owners_override_then_one_fails_gate(bridge, namespace):
    """Both owners hold overrides; o2's sender then fails. o2's retired override/baseline must not
    resolve b for o2 even though o1's override row stays active on the same subject."""
    api, ops, store, o1, o2, t1, t2, pin, a, b, key = _world(bridge, namespace)
    first = {a: (o1, True, True), b: (o2, True, True)}.get
    rebuild(bridge, sender_lookup=first)
    ops.override_key(key, pin, owner_entity_id=o1)
    ops.override_key(key, pin, owner_entity_id=o2)
    rebuild(bridge, sender_lookup=first)
    assert api.resolve_key(store, key, b, owner_entity_id=o2) == pin
    later = {a: (o1, True, True), b: (o2, True, False)}.get
    rebuild(bridge, sender_lookup=later)
    assert api.resolve_key(store, key, b, owner_entity_id=o2) is None
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin


def test_mine_item6_new_evidence_through_alias_after_override(bridge):
    """Rule (b) control: override binds only its cited row; new verified evidence resolves via alias."""
    api, ops, store, o1, o2, t1, t2, pin, a, b, key = _world(bridge, "person")
    first = {a: (o1, True, True), b: (o2, True, True)}.get
    rebuild(bridge, sender_lookup=first)
    ops.override_key(key, pin, owner_entity_id=o1)
    c = publish(bridge, key, scope="P1")
    nxt = {a: (o1, True, True), b: (o2, True, True), c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=nxt)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin
    assert api.resolve_key(store, key, c, owner_entity_id=o1) == t1
    # c then loses verification: must stop resolving even though o1's override is active.
    rebuild(bridge, sender_lookup={a: (o1, True, True), b: (o2, True, True), c: (o1, False, True)}.get)
    assert api.resolve_key(store, key, c, owner_entity_id=o1) is None
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == pin
