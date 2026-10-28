# ruff: noqa: F811  # Imported pytest fixture is intentionally injected by name.
"""Minimal repro: override before D6 gate failure, then undo without rebuild."""
import json
import pytest
from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_min_undo_revives_gate_failed_alias_row(bridge, namespace):
    api, ops, store, _, _ = bridge
    o1 = ops.create_entity("owner one")
    t1 = o1 if namespace == "user" else ops.create_entity("sister one")
    key = "user:self" if namespace == "user" else "person:sister"
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    pin = ops.create_entity("pinned")
    a = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (o1, True, True)}.get)
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    rebuild(bridge, sender_lookup={a: (o1, False, True)}.get)   # D6 gate fails: sender unverified
    assert not [r for r in active(store) if r["owner_entity_id"] == o1]   # rebuild retired the key
    ops.undo(ev)                                                  # no override remains
    rows = [(r["entity_id"], r["method"], json.loads(r["source_store_ids"])) for r in active(store)]
    got = api.resolve_key(store, key, a, owner_entity_id=o1)
    assert got is None and rows == [], f"resolved={got} t1={t1} active={rows}"


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_min_undo_then_global_override_cites_gate_failed_source(bridge, namespace):
    api, ops, store, _, _ = bridge
    o1 = ops.create_entity("owner one")
    t1 = o1 if namespace == "user" else ops.create_entity("sister one")
    key = "user:self" if namespace == "user" else "person:sister"
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    pin = ops.create_entity("pinned")
    a = publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (o1, True, True)}.get)
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    rebuild(bridge, sender_lookup={a: (o1, True, False)}.get)   # not first person
    ops.undo(ev)
    ops.override_key(key, pin, owner_entity_id=o1)
    cited = json.loads(store.events_after()[-1]["payload_json"])["override"]["scope_store_ids"]
    assert a not in cited, f"global override cites gate-failed source: {cited}"


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_min_undo_with_surviving_sibling_source(bridge, namespace):
    """Override stays active through c; a fails the gate; undo restores the pre-override row {a,c}."""
    api, ops, store, _, _ = bridge
    o1 = ops.create_entity("owner one")
    t1 = o1 if namespace == "user" else ops.create_entity("sister one")
    key = "user:self" if namespace == "user" else "person:sister"
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    pin = ops.create_entity("pinned")
    a, c = publish(bridge, key), publish(bridge, key)
    rebuild(bridge, sender_lookup={a: (o1, True, True), c: (o1, True, True)}.get)
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    later = {c: (o1, True, True)}.get                               # a's sender dropped
    rebuild(bridge, sender_lookup=later)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    ops.undo(ev)
    rows = [(r["entity_id"], r["method"], json.loads(r["source_store_ids"])) for r in active(store)]
    got = api.resolve_key(store, key, a, owner_entity_id=o1)
    assert got is None, f"resolved={got} t1={t1} active={rows}"


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_stacked_override_undo_cannot_launder_gate_failed_binding(bridge, namespace):
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("owner")
    target = owner if namespace == "user" else ops.create_entity("sister")
    key = "user:self" if namespace == "user" else "person:sister"
    ops.add_alias(target, "my sister", "relation_phrase", owner_entity_id=owner)
    chosen = ops.create_entity("chosen")
    source = publish(bridge, key)
    rebuild(bridge, sender_lookup={source: (owner, True, True)}.get)
    first = ops.override_key(key, chosen, owner_entity_id=owner)
    second = ops.override_key(key, chosen, owner_entity_id=owner)
    rebuild(bridge, sender_lookup={source: (owner, False, True)}.get)
    ops.undo(second)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) is None
    ops.undo(first)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) is None
    assert active(store) == []


def test_undo_does_not_restore_invalidated_source_without_rebuild(bridge):
    _, ops, store, _, assertions = bridge
    original, chosen = ops.create_entity("original"), ops.create_entity("chosen")
    ops.add_alias(original, "alex")
    source = publish(bridge, "person:alex")
    rebuild(bridge)
    event = ops.override_key("person:alex", chosen)
    assertions.invalidate_source(source, reason="synthetic")
    ops.undo(event)
    assert active(store) == []
