# ruff: noqa: F811
# Imported pytest fixtures are intentionally named by test arguments.
"""r1b verifier (i6-uncited) own probe: a global override binds only its cited rows."""
import json

import pytest

from tests.test_entity_bridge import active, bridge, publish, rebuild  # noqa: F401


def _setup(bridge):
    api, ops, store, _, _ = bridge
    o1 = ops.create_entity("owner")
    sister = ops.create_entity("sister")
    other = ops.create_entity("other sister")
    alias = ops.add_alias(sister, "my sister", "relation_phrase", owner_entity_id=o1)
    return api, ops, store, o1, sister, other, alias


def test_owner_override_then_two_rounds_of_new_evidence(bridge):
    api, ops, store, o1, sister, other, _ = _setup(bridge)
    owned = {}
    s1 = publish(bridge, "person:sister")
    owned[s1] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    ops.override_key("person:sister", other, owner_entity_id=o1)
    s3 = publish(bridge, "person:sister")
    owned[s3] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    s4 = publish(bridge, "person:sister")
    owned[s4] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    # idempotent second pass
    assert rebuild(bridge, sender_lookup=owned.get) == 0
    def r(s):
        return api.resolve_key(store, "person:sister", s, owner_entity_id=o1)
    assert (r(s1), r(s3), r(s4)) == (other, sister, sister)
    # the active slot still holds the override with only the cited row
    rows = [(x["method"], x["entity_id"], json.loads(x["source_store_ids"])) for x in active(store)]
    assert rows == [("override", other, [s1])]


def test_owner_override_new_evidence_follows_alias_retarget(bridge):
    api, ops, store, o1, sister, other, alias = _setup(bridge)
    sister2 = ops.create_entity("second sister")
    owned = {}
    s1 = publish(bridge, "person:sister")
    owned[s1] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    ops.override_key("person:sister", other, owner_entity_id=o1)
    s3 = publish(bridge, "person:sister")
    owned[s3] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    assert api.resolve_key(store, "person:sister", s3, owner_entity_id=o1) == sister
    ops.retire_alias(alias)
    ops.add_alias(sister2, "my sister", "relation_phrase", owner_entity_id=o1)
    rebuild(bridge, sender_lookup=owned.get)
    assert api.resolve_key(store, "person:sister", s1, owner_entity_id=o1) == other
    assert api.resolve_key(store, "person:sister", s3, owner_entity_id=o1) == sister2


def test_owner_override_alias_retired_new_evidence_unresolved(bridge):
    api, ops, store, o1, sister, other, alias = _setup(bridge)
    owned = {}
    s1 = publish(bridge, "person:sister")
    owned[s1] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    ops.override_key("person:sister", other, owner_entity_id=o1)
    s3 = publish(bridge, "person:sister")
    owned[s3] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    ops.retire_alias(alias)
    rebuild(bridge, sender_lookup=owned.get)
    assert api.resolve_key(store, "person:sister", s3, owner_entity_id=o1) is None


@pytest.mark.parametrize("with_override", [False, True])
def test_relation_key_unresolved_after_sender_verification_lost(bridge, with_override):
    """D6: a relation-phrase key resolves only with a CURRENT verified sender."""
    api, ops, store, o1, sister, other, _ = _setup(bridge)
    owned = {}
    s1 = publish(bridge, "person:sister")
    owned[s1] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    if with_override:
        ops.override_key("person:sister", other, owner_entity_id=o1)
    s3 = publish(bridge, "person:sister")
    owned[s3] = (o1, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    assert api.resolve_key(store, "person:sister", s3, owner_entity_id=o1) == sister
    owned.clear()  # no source row has a verified sender any more
    rebuild(bridge, sender_lookup=owned.get)
    print("\nactive", [(x["method"], x["entity_id"], x["source_store_ids"]) for x in active(store)])
    assert api.resolve_key(store, "person:sister", s3, owner_entity_id=o1) is None


@pytest.mark.parametrize("with_override", [False, True])
def test_user_key_unresolved_after_sender_verification_lost(bridge, with_override):
    api, ops, store, _, _, _, _ = _setup(bridge)
    speaker, alt = ops.create_entity("speaker"), ops.create_entity("alt")
    owned = {}
    s1 = publish(bridge, "user:self")
    owned[s1] = (speaker, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    if with_override:
        ops.override_key("user:self", alt, owner_entity_id=speaker)
    s3 = publish(bridge, "user:self")
    owned[s3] = (speaker, True, True)
    rebuild(bridge, sender_lookup=owned.get)
    assert api.resolve_key(store, "user:self", s3, owner_entity_id=speaker) == speaker
    owned.clear()
    rebuild(bridge, sender_lookup=owned.get)
    assert api.resolve_key(store, "user:self", s3, owner_entity_id=speaker) is None
