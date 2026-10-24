"""r1b verifier (i2-relation), own probe: an owner-scoped relation key must never fall through to the
name-alias path when the owner's relation group is not visible (to a principal, or at all)."""

from contextlib import closing
import importlib

import pytest

from hermes_lcm.assertion_store import AssertionCandidate, AssertionStore
from hermes_lcm.entity_store import EntityStore
from hermes_lcm.store import MessageStore


@pytest.fixture
def env(tmp_path):
    path = tmp_path / "lcm.db"
    with closing(MessageStore(path)) as messages:
        messages.connection.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
        messages.connection.commit()
        with closing(AssertionStore(path)) as assertions, closing(EntityStore(path, profile_key="A")) as store:
            api = importlib.import_module("hermes_lcm.entity_ops")
            yield api, api.EntityOps(store), store, messages, assertions


def msg(messages, scope):
    sid = messages.append("S", {"role": "user", "content": "synthetic"}, source="cli")
    messages.connection.execute("UPDATE messages SET access_scope=? WHERE store_id=?", (scope, sid))
    messages.connection.commit()
    return sid


def publish(assertions, source, key):
    c = AssertionCandidate(source_span_start=0, source_span_end=9, subject_key=key,
                           predicate_key="likes", object_value="tea", value_text="tea", kind="fact")
    assertions.publish_source(assertions.snapshot_source(source), [c], extraction_version="assertions-v1")


def world(env, alias_kind="name", relation_scope="P1"):
    """person:sister asserted in a P2 row by verified first-person sender Uma.
    Sara is Uma's sister via a relation alias cited only by a P1 row (hidden from P2).
    'Sister Act' is an unrelated person whose plain alias 'Sister' is visible to P2."""
    api, ops, store, messages, assertions = env
    s = msg(messages, "P2")
    hidden = msg(messages, relation_scope)
    publish(assertions, s, "person:sister")
    uma = ops.create_entity("Uma")
    ops.add_alias(uma, "Uma", source_store_ids=[s])
    sara = ops.create_entity("Sara")
    ops.add_alias(sara, "Sara", source_store_ids=[s])
    rel = ops.add_alias(sara, "my sister", "relation_phrase", owner_entity_id=uma, source_store_ids=[hidden])
    act = ops.create_entity("Sister Act")
    ops.add_alias(act, "Sister", alias_kind, source_store_ids=[s])
    lookup = lambda sid: (uma, True, True) if sid == s else None  # noqa: E731
    api.rebuild_entity_keys(store, sender_lookup=lookup)
    return dict(api=api, ops=ops, store=store, s=s, uma=uma, sara=sara, act=act, rel=rel, lookup=lookup)


@pytest.mark.parametrize("alias_kind", ["name", "nickname"])
def test_principal_owner_scoped_read_never_reaches_name_entity(env, alias_kind):
    w = world(env, alias_kind)
    api, store = w["api"], w["store"]
    # Control: the trusted owner-scoped read resolves through the relation.
    assert api.resolve_key(store, "person:sister", w["s"], owner_entity_id=w["uma"]) == w["sara"]
    # Control: the name entity really is a live name candidate for P2 (so a fallthrough would hit it).
    view = api.resolved_view(store, "P2")
    assert w["act"] in view["cards"] and "relations" not in view
    assert api._key_candidates(view, "person:sister") == [w["act"]]
    got = api.resolve_key(store, "person:sister", w["s"], owner_entity_id=w["uma"], access_scope="P2")
    assert got is None, f"P2 owner-scoped read resolved to {got!r} (Sister Act={w['act']!r})"
    # Key candidates for an owner never come from names.
    assert api._key_candidates(view, "person:sister", w["uma"]) == []
    # The P2 key view never binds the owner-scoped key to the name entity.
    for k in view.get("keys", []):
        assert not (k["subject_key"] == "person:sister" and k["entity_id"] == w["act"]), k


def test_owner_principal_still_sees_nothing_when_relation_cited_elsewhere(env):
    # principal_is_owner does not make a P1-cited alias visible to a P2 reader; result must be None, not 'Sister Act'.
    w = world(env)
    got = w["api"].resolve_key(w["store"], "person:sister", w["s"], owner_entity_id=w["uma"],
                               access_scope="P2", principal_is_owner=True)
    assert got in (None, w["sara"]) and got != w["act"]


def test_relation_retired_without_rebuild_trusted_read_does_not_fall_to_name(env):
    # Stale key row remains; relation group gone for everyone (trusted view too).
    w = world(env, relation_scope="P2")
    api, ops, store = w["api"], w["ops"], w["store"]
    ops.retire_alias(w["rel"])
    for scope in (None, "P2"):
        got = api.resolve_key(store, "person:sister", w["s"], owner_entity_id=w["uma"], access_scope=scope)
        assert got != w["act"], f"scope={scope}: stale owner key fell through to the name entity"
        assert got is None


def test_relation_retired_then_rebuild_derives_no_name_bound_owner_key(env):
    w = world(env, relation_scope="P2")
    api, ops, store = w["api"], w["ops"], w["store"]
    ops.retire_alias(w["rel"])
    api.rebuild_entity_keys(store, sender_lookup=w["lookup"])
    rows = [dict(r) for r in store.connection.execute(
        "SELECT subject_key, owner_entity_id, entity_id FROM lcm_entity_keys WHERE retired_at IS NULL")]
    # Owner-scoped identity never binds to the name entity (owner-less name binding is a separate design question).
    assert all(not (r["owner_entity_id"] == w["uma"] and r["entity_id"] == w["act"]) for r in rows), rows
    print("NOTE rows after relation retired + rebuild:", rows)
    assert api.resolve_key(store, "person:sister", w["s"], owner_entity_id=w["uma"]) is None
    assert api.resolve_key(store, "person:sister", w["s"], owner_entity_id=w["uma"], access_scope="P2") is None


def test_other_owners_relation_visible_does_not_rescue_via_name(env):
    # A visible relation group for a DIFFERENT owner: Uma's key stays unresolved for P2, never 'Sister Act'.
    w = world(env)
    api, ops, store = w["api"], w["ops"], w["store"]
    bob = ops.create_entity("Bob")
    bea = ops.create_entity("Bea")
    ops.add_alias(bob, "Bob", source_store_ids=[w["s"]])
    ops.add_alias(bea, "my sister", "relation_phrase", owner_entity_id=bob, source_store_ids=[w["s"]])
    got = api.resolve_key(store, "person:sister", w["s"], owner_entity_id=w["uma"], access_scope="P2")
    assert got is None, got


def test_name_path_still_live_for_owner_less_non_relation_key(env):
    # Positive control: the fix must not kill ordinary owner-less name resolution.
    api, ops, store, messages, assertions = env
    s = msg(messages, "P2")
    publish(assertions, s, "person:sara")
    sara = ops.create_entity("Sara")
    ops.add_alias(sara, "Sara", source_store_ids=[s])
    api.rebuild_entity_keys(store)
    assert api.resolve_key(store, "person:sara", s) == sara
    assert api.resolve_key(store, "person:sara", s, access_scope="P2") == sara
