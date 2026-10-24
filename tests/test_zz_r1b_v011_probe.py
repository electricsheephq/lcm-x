"""Verifier 2 (1926-v0-1-1): owner-scoped relation key falling through to name aliases for a principal."""

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


def message(messages, scope):
    sid = messages.append("S", {"role": "user", "content": "synthetic"}, source="cli")
    messages.connection.execute("UPDATE messages SET access_scope=? WHERE store_id=?", (scope, sid))
    messages.connection.commit()
    return sid


def assertion(assertions, source, key):
    c = AssertionCandidate(source_span_start=0, source_span_end=9, subject_key=key,
                           predicate_key="likes", object_value="tea", value_text="tea", kind="fact")
    assertions.publish_source(assertions.snapshot_source(source), [c], extraction_version="assertions-v1")


def setup(env, relation_cited):
    api, ops, store, messages, assertions = env
    s = message(messages, "P2")
    p1_only = message(messages, "P1")
    assertion(assertions, s, "person:mom")
    uma = ops.create_entity("Uma")
    ops.add_alias(uma, "Uma", source_store_ids=[s])
    martha = ops.create_entity("Martha")
    ops.add_alias(martha, "Martha", source_store_ids=[s])
    if relation_cited:
        ops.add_alias(martha, "my mom", "relation_phrase", owner_entity_id=uma, source_store_ids=[p1_only])
    else:
        ops.add_alias(martha, "my mom", "relation_phrase", owner_entity_id=uma)
    jones = ops.create_entity("Mom Jones")
    ops.add_alias(jones, "Mom", "nickname", source_store_ids=[s])
    api.rebuild_entity_keys(store, sender_lookup=lambda sid: (uma, True, True) if sid == s else None)
    return api, store, s, uma, martha, jones


@pytest.mark.parametrize("relation_cited", [True, False])
def test_zz_r1b_v011_owner_scoped_key_principal_read(env, relation_cited):
    api, store, s, uma, martha, jones = setup(env, relation_cited)
    rows = [dict(r) for r in store.connection.execute(
        "SELECT subject_key, owner_entity_id, entity_id, method FROM lcm_entity_keys WHERE retired_at IS NULL")]
    print("stored keys", rows)
    assert rows == [dict(subject_key="person:mom", owner_entity_id=uma, entity_id=martha, method="alias_match")]
    # Controls: trusted owner-scoped read resolves to Martha.
    assert api.resolve_key(store, "person:mom", s, owner_entity_id=uma) == martha
    # P2 view of keys says (person:mom, U1) -> Martha.
    view = api.resolved_view(store, "P2")
    print("P2 view keys", view.get("keys"), "relations", view.get("relations"))
    # r1b adaptation: item 7 makes P2's key view drop a binding whose only support is hidden.
    assert [(k["subject_key"], k["owner_entity_id"], k["entity_id"]) for k in view.get("keys", [])] in (
        [], [("person:mom", uma, martha)])
    assert jones in view["cards"] and martha in view["cards"]
    got = api.resolve_key(store, "person:mom", s, owner_entity_id=uma, access_scope="P2", principal_is_owner=False)
    print("P2 resolve_key ->", got, "martha", martha, "jones", jones)
    assert got != jones, "owner-scoped key resolved, for principal P2, to the unrelated name/nickname entity"
    assert got in (martha, None)


def test_zz_r1b_v011_trusted_vs_principal_owner_none(env):
    api, store, s, uma, martha, jones = setup(env, True)
    trusted = api.resolve_key(store, "person:mom", s)
    principal = api.resolve_key(store, "person:mom", s, access_scope="P2", principal_is_owner=False)
    print("owner=None trusted", trusted, "principal", principal, "jones", jones)


def test_zz_r1b_v011_owner_principal_sees_relation_control(env):
    # principal_is_owner=True makes an uncited relation alias visible: should then resolve to Martha.
    api, store, s, uma, martha, jones = setup(env, False)
    assert api.resolve_key(store, "person:mom", s, owner_entity_id=uma, access_scope="P2",
                           principal_is_owner=True) == martha


@pytest.mark.parametrize("relation_cited", [True, False])
def test_zz_r1b_v011_candidate_fix_causality(env, relation_cited, monkeypatch):
    api, store, s, uma, martha, jones = setup(env, relation_cited)
    original = api._key_candidates

    def patched(view, subject_key, owner=None):
        namespace, _, value = subject_key.partition(":")
        if owner is not None and namespace not in {"user", "assistant"} and not api._relation_groups(view, value):
            return []
        return original(view, subject_key, owner)

    monkeypatch.setattr(api, "_key_candidates", patched)
    got = api.resolve_key(store, "person:mom", s, owner_entity_id=uma, access_scope="P2", principal_is_owner=False)
    assert got is None
    assert api.resolve_key(store, "person:mom", s, owner_entity_id=uma) == martha
