# ruff: noqa: F811
# Imported pytest fixtures are intentionally named by test arguments.
"""Independent probe (i6-uncited-c): retired alias_match rows must not resolve after the owner derivation is gone."""
import pytest
from tests.test_entity_bridge import bridge, publish, rebuild, active  # noqa: F401


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("loss", ["unverified", "not_first_person", "other_owner"])
def test_no_override_stale_owner_row_does_not_resolve(bridge, namespace, loss):
    api, ops, store, _, _ = bridge
    o1, o2, sister = ops.create_entity("Owner"), ops.create_entity("Other"), ops.create_entity("Sister")
    ops.add_alias(sister, "my sister", "relation_phrase", owner_entity_id=o1)
    subject = namespace + (":self" if namespace == "user" else ":sister")
    expected = o1 if namespace == "user" else sister
    src = publish(bridge, subject)
    rebuild(bridge, sender_lookup=lambda sid: (o1, True, True))
    # positive control: derivation present
    assert api.resolve_key(store, subject, src, owner_entity_id=o1) == expected
    lost = {"unverified": (o1, False, True), "not_first_person": (o1, True, False),
            "other_owner": (o2, True, True)}[loss]
    rebuild(bridge, sender_lookup=lambda sid: lost)
    assert not any(r["owner_entity_id"] == o1 for r in active(store))
    # D6: with no verified first-person sender of o1, the o1 key must be unresolved
    assert api.resolve_key(store, subject, src, owner_entity_id=o1) is None
