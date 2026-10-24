# ruff: noqa: F811
# Imported pytest fixtures are intentionally named by test arguments.
"""Independent probe (i5-owners-c): retired alias_match rows must not act as live owner evidence."""
import json
import pytest
from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401


def _setup(bridge):
    api, ops, store, _, _ = bridge
    o1, o2 = ops.create_entity("speaker one"), ops.create_entity("speaker two")
    s1, s2 = ops.create_entity("sister one"), ops.create_entity("sister two")
    ops.add_alias(s1, "sister", "relation_phrase", owner_entity_id=o1)
    ops.add_alias(s2, "sister", "relation_phrase", owner_entity_id=o2)
    a, b = publish(bridge, "person:sister"), publish(bridge, "person:sister")
    rebuild(bridge, sender_lookup={a: (o1, True, True), b: (o2, True, True)}.get)
    return api, ops, store, o1, o2, s1, s2, a, b


def test_control_resolves_for_current_owner(bridge):
    api, ops, store, o1, o2, s1, s2, a, b = _setup(bridge)
    assert api.resolve_key(store, "person:sister", a, owner_entity_id=o1) == s1
    assert api.resolve_key(store, "person:sister", b, owner_entity_id=o2) == s2


@pytest.mark.parametrize("variant", ["unverified", "moved", "dropped"])
def test_retired_owner_row_never_resolves(bridge, variant):
    api, ops, store, o1, o2, s1, s2, a, b = _setup(bridge)
    lookup = {"unverified": {a: (o1, False, True), b: (o2, True, True)},
              "moved": {a: (o2, True, True), b: (o2, True, True)},
              "dropped": {b: (o2, True, True)}}[variant]
    rebuild(bridge, sender_lookup=lookup.get)
    # o1 has no active key row any more
    assert not [r for r in active(store) if r["owner_entity_id"] == o1]
    got = api.resolve_key(store, "person:sister", a, owner_entity_id=o1)
    assert got is None, f"{variant}: retired o1 row still resolves a -> {got} (s1={s1})"


def test_override_after_retire_cites_no_stale_source(bridge):
    api, ops, store, o1, o2, s1, s2, a, b = _setup(bridge)
    rebuild(bridge, sender_lookup={a: (o1, False, True), b: (o2, True, True)}.get)
    pin = ops.create_entity("pinned", "person")
    ops.override_key("person:sister", pin, owner_entity_id=o1)
    rows = [r for r in active(store) if r["owner_entity_id"] == o1]
    cited = [json.loads(r["source_store_ids"]) for r in rows]
    assert cited in ([], [[]]), f"o1 override cites stale sources {cited} (a={a})"
    rebuild(bridge, sender_lookup={a: (o1, False, True), b: (o2, True, True)}.get)
    assert api.resolve_key(store, "person:sister", a, owner_entity_id=o1) != pin
