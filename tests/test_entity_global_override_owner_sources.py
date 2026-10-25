# ruff: noqa: F811
# Imported pytest fixtures are intentionally named by test arguments.
"""i5-owners reviewer probes (own, written from the finding text): an owner-scoped global
override must cite (and bind) only THAT owner's CURRENT sources of the key."""

import json

from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401


def _keys(store):
    return {(r["subject_key"], r["owner_entity_id"]): (r["entity_id"], r["method"], json.loads(r["source_store_ids"]))
            for r in active(store)}


def _setup(bridge):
    api, ops, store, _, _ = bridge
    o1, o2 = ops.create_entity("owner one"), ops.create_entity("owner two")
    sis1, sis2 = ops.create_entity("sister one"), ops.create_entity("sister two")
    pin = ops.create_entity("pinned sister")
    a = publish(bridge, "person:sister", scope="P1")
    b = publish(bridge, "person:sister", scope="P2")
    ops.add_alias(sis1, "my sister", "relation_phrase", owner_entity_id=o1)
    ops.add_alias(sis2, "my sister", "relation_phrase", owner_entity_id=o2)
    return api, ops, store, o1, o2, sis1, sis2, pin, a, b


def test_own_a_three_owners_override_cites_only_own(bridge):
    """Three owners, two sources each; o2's override cites exactly o2's two sources."""
    api, ops, store, _, _ = bridge
    owners = [ops.create_entity(f"owner {i}") for i in range(3)]
    sisters = [ops.create_entity(f"sister {i}") for i in range(3)]
    pin = ops.create_entity("pinned")
    src = {o: [publish(bridge, "person:sister", scope=f"P{i}") for _ in range(2)] for i, o in enumerate(owners)}
    for o, s in zip(owners, sisters):
        ops.add_alias(s, "my sister", "relation_phrase", owner_entity_id=o)
    senders = {sid: (o, True, True) for o, sids in src.items() for sid in sids}.get
    rebuild(bridge, sender_lookup=senders)
    assert sorted(_keys(store)[("person:sister", owners[1])][2]) == sorted(src[owners[1]])
    ops.override_key("person:sister", pin, owner_entity_id=owners[1])
    payload = json.loads(store.events_after()[-1]["payload_json"])["override"]
    assert payload["scope_store_ids"] == sorted(src[owners[1]])
    rebuild(bridge, sender_lookup=senders)
    keys = _keys(store)
    assert keys[("person:sister", owners[1])] == (pin, "override", sorted(src[owners[1]]))
    for o, s in ((owners[0], sisters[0]), (owners[2], sisters[2])):
        assert keys[("person:sister", o)] == (s, "alias_match", sorted(src[o]))
    claims = {}
    for (_, o), (_, _, cited) in keys.items():
        for sid in cited:
            claims.setdefault(sid, []).append(o)
    assert all(len(v) == 1 for v in claims.values()), claims


def test_own_b_override_after_reattribution_cites_only_current(bridge):
    """s=a moves from o1 to o2 (sender re-attributed) and a rebuild runs; o1's later
    global override must not cite a (no longer o1's current source)."""
    api, ops, store, o1, o2, sis1, sis2, pin, a, b = _setup(bridge)
    c = publish(bridge, "person:sister", scope="P1")
    first = {a: (o1, True, True), b: (o2, True, True), c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=first)
    assert _keys(store)[("person:sister", o1)] == (sis1, "alias_match", sorted([a, c]))
    moved = {a: (o2, True, True), b: (o2, True, True), c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=moved)
    before = _keys(store)
    print("after move:", before)
    assert before[("person:sister", o1)] == (sis1, "alias_match", [c])
    assert before[("person:sister", o2)] == (sis2, "alias_match", sorted([a, b]))
    ops.override_key("person:sister", pin, owner_entity_id=o1)
    payload = json.loads(store.events_after()[-1]["payload_json"])["override"]
    print("payload:", payload)
    assert payload["scope_store_ids"] == [c], "override cites a source that is now o2's"


def test_own_c_override_after_owner_lost_all_sources(bridge):
    """o1's only source loses its verified sender; after a rebuild o1 has no current
    sources, so a global override cites nothing (and binds nothing)."""
    api, ops, store, o1, o2, sis1, sis2, pin, a, b = _setup(bridge)
    rebuild(bridge, sender_lookup={a: (o1, True, True), b: (o2, True, True)}.get)
    rebuild(bridge, sender_lookup={b: (o2, True, True)}.get)
    assert ("person:sister", o1) not in _keys(store)
    try:
        ops.override_key("person:sister", pin, owner_entity_id=o1)
    except api.EntityOpRefused as exc:
        print("refused:", exc)
        return
    payload = json.loads(store.events_after()[-1]["payload_json"])["override"]
    print("payload:", payload)
    assert a not in payload["scope_store_ids"], "override cites o1's stale (now unattributed) source"
    rebuild(bridge, sender_lookup={b: (o2, True, True)}.get)
    keys = _keys(store)
    print("keys:", keys)
    assert a not in keys.get(("person:sister", o1), (None, None, []))[2]


def test_own_d_rebuild_drops_reattributed_source_from_override_binding(bridge):
    """ADR D6: a cited row applies 'while that row is still a current source of the key'.
    After a is re-attributed to o2, o1's override row must stop binding a; a is never
    claimed by two owner keys."""
    api, ops, store, o1, o2, sis1, sis2, pin, a, b = _setup(bridge)
    c = publish(bridge, "person:sister", scope="P1")
    first = {a: (o1, True, True), b: (o2, True, True), c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=first)
    ops.override_key("person:sister", pin, owner_entity_id=o1)
    assert json.loads(store.events_after()[-1]["payload_json"])["override"]["scope_store_ids"] == sorted([a, c])
    moved = {a: (o2, True, True), b: (o2, True, True), c: (o1, True, True)}.get
    rebuild(bridge, sender_lookup=moved)
    keys = _keys(store)
    print("keys after move:", keys)
    claimants = sorted(o for (_, o), (_, _, cited) in keys.items() if a in cited)
    print("claimants of a:", claimants, "o1", o1, "o2", o2)
    print("resolve a as o1:", api.resolve_key(store, "person:sister", a, owner_entity_id=o1),
          "as o2:", api.resolve_key(store, "person:sister", a, owner_entity_id=o2))
    assert claimants == [o2], "o1's override still binds a source that is now o2's"


def test_own_e_stale_retired_row_cited_and_bound_after_rebuild(bridge):
    """Consequence of own_c: the stale cite is bound by rebuild, so an unverified relation-phrase
    source (D6: stays unresolved) resolves through o1's override."""
    api, ops, store, o1, o2, sis1, sis2, pin, a, b = _setup(bridge)
    rebuild(bridge, sender_lookup={a: (o1, True, True), b: (o2, True, True)}.get)
    later = {b: (o2, True, True)}.get
    rebuild(bridge, sender_lookup=later)
    assert api.resolve_key(store, "person:sister", a, owner_entity_id=o1) is None  # control
    ops.override_key("person:sister", pin, owner_entity_id=o1)
    rebuild(bridge, sender_lookup=later)
    keys = _keys(store)
    print("keys:", keys)
    assert api.resolve_key(store, "person:sister", a, owner_entity_id=o1) is None, \
        "unverified source a resolves via o1's override after it lost its sender"


import pytest


@pytest.mark.parametrize("change", ["unverified", "moved"])
def test_own_f_retired_alias_row_never_resolves(bridge, change):
    """No override at all. o1's only source a loses its verified sender (or moves to o2) and a
    rebuild retires o1's key row. resolve_key(a, owner=o1) must not resolve through the retired row."""
    api, ops, store, o1, o2, sis1, sis2, pin, a, b = _setup(bridge)
    rebuild(bridge, sender_lookup={a: (o1, True, True), b: (o2, True, True)}.get)
    assert api.resolve_key(store, "person:sister", a, owner_entity_id=o1) == sis1  # positive control
    later = {a: (o2, True, True), b: (o2, True, True)} if change == "moved" else {b: (o2, True, True)}
    rebuild(bridge, sender_lookup=later.get)
    assert ("person:sister", o1) not in _keys(store)  # o1's key row is retired
    assert api.resolve_key(store, "person:sister", a, owner_entity_id=o1) is None
