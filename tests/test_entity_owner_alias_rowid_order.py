# ruff: noqa: F811
"""guard-c independent probe: undo + re-override vs _owner_alias_rows rowid ordering."""
import json

import pytest

from tests.test_entity_bridge import bridge, publish, rebuild  # noqa: F401


def _rows(store):
    return [(r["rid"], r["method"], r["entity_id"], r["retired_at"] is None, r["source_store_ids"])
            for r in store.connection.execute(
                "SELECT rowid AS rid, * FROM lcm_entity_keys WHERE scope_key=? ORDER BY rowid", (store.scope_key,))]


def _setup(bridge, namespace):
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("owner one")
    target = owner if namespace == "user" else ops.create_entity("sister one")
    key = "user:self" if namespace == "user" else "person:sister"
    ops.add_alias(target, "my sister", "relation_phrase", owner_entity_id=owner)
    pin = ops.create_entity("pinned")
    return api, ops, store, owner, target, key, pin


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("b_state", ["unverified", "not_first_person", "dropped"])
def test_guard_c_undo_then_reoverride_no_stale(bridge, namespace, b_state):
    api, ops, store, owner, target, key, pin = _setup(bridge, namespace)
    a = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (owner, True, True)}.get)
    assert api.resolve_key(store, key, a, owner_entity_id=owner) == target
    ops.override_key(key, pin, owner_entity_id=owner)
    ev = store.events_after()[-1]["event_id"]
    b = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (owner, True, True), b: (owner, True, True)}.get)
    assert api.resolve_key(store, key, b, owner_entity_id=owner) == target  # item 6
    ops.undo(ev)
    bad = {"unverified": (owner, False, True), "not_first_person": (owner, True, False)}.get(b_state)
    lookup = {a: (owner, True, True)}
    if bad:
        lookup[b] = bad
    rebuild(bridge, sender_lookup=lookup.get)
    assert api.resolve_key(store, key, b, owner_entity_id=owner) is None  # b fails D6 now
    ops.override_key(key, pin, owner_entity_id=owner)
    cited = json.loads(store.events_after()[-1]["payload_json"])["override"]["scope_store_ids"]
    print(_rows(store))
    assert cited == [a]
    assert api.resolve_key(store, key, a, owner_entity_id=owner) == pin
    assert api.resolve_key(store, key, b, owner_entity_id=owner) is None, _rows(store)


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_guard_c_stacked_override_after_undo_cites_only_passing(bridge, namespace):
    api, ops, store, owner, target, key, pin = _setup(bridge, namespace)
    pin2 = ops.create_entity("pinned two")
    a = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (owner, True, True)}.get)
    ops.override_key(key, pin, owner_entity_id=owner)
    ev = store.events_after()[-1]["event_id"]
    b = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (owner, True, True), b: (owner, True, True)}.get)
    ops.undo(ev)
    rebuild(bridge, sender_lookup={a: (owner, True, True), b: (owner, False, True)}.get)
    ops.override_key(key, pin, owner_entity_id=owner)
    ops.override_key(key, pin2, owner_entity_id=owner)
    cited = json.loads(store.events_after()[-1]["payload_json"])["override"]["scope_store_ids"]
    assert b not in cited, (cited, _rows(store))


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_guard_c_control_without_undo(bridge, namespace):
    """Positive control: same final sender map, no undo -> b unresolved after re-override."""
    api, ops, store, owner, target, key, pin = _setup(bridge, namespace)
    a = publish(bridge, key)
    b = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (owner, True, True), b: (owner, False, True)}.get)
    ops.override_key(key, pin, owner_entity_id=owner)
    assert api.resolve_key(store, key, b, owner_entity_id=owner) is None
    assert api.resolve_key(store, key, a, owner_entity_id=owner) == pin


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_guard_c_one_rebuild_repairs(bridge, namespace):
    api, ops, store, owner, target, key, pin = _setup(bridge, namespace)
    a = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (owner, True, True)}.get)
    ops.override_key(key, pin, owner_entity_id=owner)
    ev = store.events_after()[-1]["event_id"]
    b = publish(bridge, key)
    lk_ok = {a: (owner, True, True), b: (owner, True, True)}.get
    rebuild(bridge, sender_lookup=lk_ok)
    ops.undo(ev)
    lk_bad = {a: (owner, True, True), b: (owner, False, True)}.get
    rebuild(bridge, sender_lookup=lk_bad)
    ops.override_key(key, pin, owner_entity_id=owner)
    rebuild(bridge, sender_lookup=lk_bad)
    assert api.resolve_key(store, key, b, owner_entity_id=owner) is None
