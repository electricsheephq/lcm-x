# ruff: noqa: F811
# Imported pytest fixtures are intentionally named by test arguments.
"""Independent probe (g-regress-c): a derivation retired because D6 gates failed must not resolve."""
import pytest
from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("verified,first_person", [(False, True), (True, False), (False, False)])
def test_zz_r1b_c_gate_failure_after_verified_run(bridge, namespace, verified, first_person):
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("Owner")
    target = owner if namespace == "user" else ops.create_entity("Sister")
    ops.add_alias(target, "my sister", "relation_phrase", owner_entity_id=owner)
    key = "user:self" if namespace == "user" else "person:sister"
    src = publish(bridge, key)
    rebuild(bridge, sender_lookup=lambda s: (owner, True, True))
    assert api.resolve_key(store, key, src, owner_entity_id=owner) == target  # positive control
    rebuild(bridge, sender_lookup=lambda s: (owner, verified, first_person))
    assert active(store) == []
    assert api.resolve_key(store, key, src, owner_entity_id=owner) is None


def test_zz_r1b_c_sender_moves_to_other_owner(bridge):
    api, ops, store, _, _ = bridge
    a, b = ops.create_entity("A"), ops.create_entity("B")
    sa, sb = ops.create_entity("SisA"), ops.create_entity("SisB")
    ops.add_alias(sa, "my sister", "relation_phrase", owner_entity_id=a)
    ops.add_alias(sb, "my sister", "relation_phrase", owner_entity_id=b)
    src = publish(bridge, "person:sister")
    rebuild(bridge, sender_lookup=lambda s: (a, True, True))
    assert api.resolve_key(store, "person:sister", src, owner_entity_id=a) == sa
    rebuild(bridge, sender_lookup=lambda s: (b, True, True))
    assert api.resolve_key(store, "person:sister", src, owner_entity_id=b) == sb
    # A no longer owns this source's speech; its retired derivation must not answer.
    assert api.resolve_key(store, "person:sister", src, owner_entity_id=a) is None
