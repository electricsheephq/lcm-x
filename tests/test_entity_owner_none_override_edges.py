# ruff: noqa: F811
"""U1 edge probes: owner-None override on a key that later becomes a relation key; key_held."""
import json
from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401


def _rows(store):
    return [(r["subject_key"], r["owner_entity_id"], r["entity_id"], r["method"], json.loads(r["source_store_ids"]))
            for r in active(store)]


def test_edge_relation_added_after_unowned_override_then_undo(bridge):
    api, ops, store, _, _ = bridge
    s = ops.create_entity("Sis")
    ops.add_alias(s, "sister")
    o1 = ops.create_entity("owner one")
    t1 = ops.create_entity("sister one")
    pin = ops.create_entity("pinned")
    a = publish(bridge, "person:sister")
    rebuild(bridge)
    assert _rows(store) == [("person:sister", None, s, "alias_match", [a])]
    ev = ops.override_key("person:sister", pin)
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    gate = {a: (o1, False, True)}.get
    rebuild(bridge, sender_lookup=gate)
    pre = (_rows(store), api.resolve_key(store, "person:sister", a))
    ops.undo(ev)
    post = (_rows(store), api.resolve_key(store, "person:sister", a),
            api.resolve_key(store, "person:sister", a, owner_entity_id=o1))
    rebuild(bridge, sender_lookup=gate)
    after = (_rows(store), api.resolve_key(store, "person:sister", a))
    print("PRE", pre, "\nPOST", post, "\nAFTER-REBUILD", after)
    assert post[1] is None and post[2] is None


def test_edge_key_held_after_gate_fail_and_new_source(bridge):
    api, ops, store, _, _ = bridge
    o1 = ops.create_entity("owner one")
    t1 = ops.create_entity("sister one")
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    pin = ops.create_entity("pinned")
    a = publish(bridge, "person:sister")
    rebuild(bridge, sender_lookup={a: (o1, True, True)}.get)
    ev = ops.override_key("person:sister", pin, owner_entity_id=o1)
    c = publish(bridge, "person:sister")
    rebuild(bridge, sender_lookup={a: (o1, False, True), c: (o1, True, True)}.get)
    print("ROWS", _rows(store))
    try:
        ops.undo(ev)
        print("UNDO OK", _rows(store))
    except api.EntityOpRefused as e:
        print("UNDO REFUSED", e.reason)
    assert api.resolve_key(store, "person:sister", a, owner_entity_id=o1) is None
    assert a not in sum((r[4] for r in _rows(store)), [])
