"""Regression-lens probes for #1926 r1b (reviewer-authored, not lane tests)."""

from contextlib import closing
import importlib
import json

import pytest

from hermes_lcm.assertion_store import AssertionCandidate, AssertionStore
from hermes_lcm.entity_store import EntityStore
from hermes_lcm.store import MessageStore


@pytest.fixture
def bridge(tmp_path):
    path = tmp_path / "lcm.db"
    with closing(MessageStore(path)) as messages:
        messages.connection.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
        messages.connection.commit()
        with closing(AssertionStore(path)) as assertions, closing(EntityStore(path, profile_key="A")) as store:
            api = importlib.import_module("hermes_lcm.entity_ops")
            yield api, api.EntityOps(store), store, messages, assertions


def publish(bridge, key, scope="P1"):
    _, _, _, messages, assertions = bridge
    source = messages.append("S", {"role": "user", "content": "synthetic"}, source="cli")
    messages.connection.execute("UPDATE messages SET access_scope=? WHERE store_id=?", (scope, source))
    messages.connection.commit()
    candidate = AssertionCandidate(source_span_start=0, source_span_end=9, subject_key=key,
        predicate_key="likes", object_value="tea", value_text="tea", kind="fact")
    assertions.publish_source(assertions.snapshot_source(source), [candidate], extraction_version="assertions-v1")
    return source


def active(store):
    return [dict(r) for r in store.connection.execute(
        "SELECT * FROM lcm_entity_keys WHERE scope_key=? AND retired_at IS NULL ORDER BY key_id", (store.scope_key,))]


def test_zz_r1b_retired_user_key_does_not_resolve(bridge):
    """A user key the derivation retired (sender no longer verified) must stay unresolved."""
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("Owner")
    ops.add_alias(owner, "owner")
    source = publish(bridge, "user:self")
    api.rebuild_entity_keys(store, sender_lookup=lambda sid: (owner, True, True))
    assert api.resolve_key(store, "user:self", source, owner_entity_id=owner) == owner
    api.rebuild_entity_keys(store, sender_lookup=lambda sid: (owner, False, True))
    assert active(store) == []
    assert api.resolve_key(store, "user:self", source, owner_entity_id=owner) is None


def test_zz_r1b_hidden_cited_user_key_principal(bridge):
    """Principal read of a user key whose stored row also cites a hidden source (D8 letter)."""
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("Owner")
    s1, s2 = publish(bridge, "user:self", "P1"), publish(bridge, "user:self", "P2")
    ops.add_alias(owner, "owner", source_store_ids=[s1])
    api.rebuild_entity_keys(store, sender_lookup=lambda sid: (owner, True, True))
    assert sorted(json.loads(active(store)[0]["source_store_ids"])) == sorted([s1, s2])
    assert api.resolved_view(store, "P1").get("keys", []) == []
    print("PRINCIPAL_RESOLVE", api.resolve_key(store, "user:self", s1, owner_entity_id=owner, access_scope="P1"))


def test_zz_r1b_rebuild_retires_never_deletes_and_is_idempotent(bridge):
    api, ops, store, messages, _ = bridge
    owner, sister, chosen = [ops.create_entity(n) for n in ("Owner", "Sister", "Chosen")]
    ops.add_alias(sister, "my sister", "relation_phrase", owner_entity_id=owner)
    first = publish(bridge, "person:sister")
    look = lambda sid: (owner, True, True)  # noqa: E731
    api.rebuild_entity_keys(store, sender_lookup=look)
    ops.override_key("person:sister", chosen, owner_entity_id=owner)
    count = lambda: store.connection.execute("SELECT count(*) FROM lcm_entity_keys").fetchone()[0]  # noqa: E731
    for _ in range(3):
        publish(bridge, "person:sister")
        before = count()
        api.rebuild_entity_keys(store, sender_lookup=look)
        assert count() >= before
        assert api.rebuild_entity_keys(store, sender_lookup=look) == 0
    assert len(active(store)) == 1 and active(store)[0]["method"] == "override"
    assert api.resolve_key(store, "person:sister", first, owner_entity_id=owner) == chosen
    messages.delete_session_messages("S")
    before = count()
    api.rebuild_entity_keys(store, sender_lookup=look)
    assert count() == before and active(store) == []


def test_zz_r1b_undo_owner_override_does_not_retire_other_owner_row(bridge):
    api, ops, store, _, _ = bridge
    owners = [ops.create_entity("Owner") for _ in range(2)]
    sisters = [ops.create_entity("Sister") for _ in range(2)]
    chosens = [ops.create_entity("Chosen", "other") for _ in range(2)]
    for o, s in zip(owners, sisters):
        ops.add_alias(s, "my sister", "relation_phrase", owner_entity_id=o)
    srcs = [publish(bridge, "person:sister") for _ in owners]
    look = dict(zip(srcs, [(o, True, True) for o in owners])).get
    api.rebuild_entity_keys(store, sender_lookup=look)
    e0 = ops.override_key("person:sister", chosens[0], owner_entity_id=owners[0])
    ops.override_key("person:sister", chosens[1], owner_entity_id=owners[1])
    api.rebuild_entity_keys(store, sender_lookup=look)
    ops.undo(e0)
    rows = {r["owner_entity_id"]: r["method"] for r in active(store)}
    assert rows == {owners[0]: "alias_match", owners[1]: "override"}
    api.rebuild_entity_keys(store, sender_lookup=look)
    assert api.resolve_key(store, "person:sister", srcs[0], owner_entity_id=owners[0]) == sisters[0]
    assert api.resolve_key(store, "person:sister", srcs[1], owner_entity_id=owners[1]) == chosens[1]


def test_zz_r1b_kind_refusals_are_non_mutating(bridge):
    api, ops, store, _, _ = bridge
    org = ops.create_entity("Acme", "org")
    publish(bridge, "person:acme")
    before = list(store.connection.iterdump())
    for subject, reason in [("person:acme", "kind_mismatch"), ("event:acme", "unbridged_namespace")]:
        with pytest.raises(api.EntityOpRefused, match=reason):
            ops.override_key(subject, org)
    assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("verified,first_person", [(False, True), (True, False)])
def test_zz_r1b_rejected_speech_after_flip_stays_unresolved(bridge, namespace, verified, first_person):
    """D6: unverified sender or non-first-person phrasing leaves the key unresolved."""
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("Owner")
    target = owner if namespace == "user" else ops.create_entity("Sister")
    ops.add_alias(target, "my sister", "relation_phrase", owner_entity_id=owner)
    subject = namespace + (":self" if namespace == "user" else ":sister")
    source = publish(bridge, subject)
    api.rebuild_entity_keys(store, sender_lookup=lambda sid: (owner, True, True))
    assert api.resolve_key(store, subject, source, owner_entity_id=owner) == target
    api.rebuild_entity_keys(store, sender_lookup=lambda sid: (owner, verified, first_person))
    assert active(store) == []
    assert api.resolve_key(store, subject, source, owner_entity_id=owner) is None
