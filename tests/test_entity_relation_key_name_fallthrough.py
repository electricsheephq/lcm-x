"""Independent verifier probe (1926-v0-1-0): owner-scoped relation key falling through to name candidates."""

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


def message(env, scope):
    _, _, _, messages, _ = env
    sid = messages.append("S", {"role": "user", "content": "synthetic"}, source="cli")
    messages.connection.execute("UPDATE messages SET access_scope=? WHERE store_id=?", (scope, sid))
    messages.connection.commit()
    return sid


def publish(env, key, source):
    assertions = env[4]
    candidate = AssertionCandidate(source_span_start=0, source_span_end=9, subject_key=key,
        predicate_key="likes", object_value="tea", value_text="tea", kind="fact")
    assertions.publish_source(assertions.snapshot_source(source), [candidate], extraction_version="assertions-v1")


def active(store):
    return [(r["subject_key"], r["owner_entity_id"], r["entity_id"], r["method"]) for r in store.connection.execute(
        "SELECT * FROM lcm_entity_keys WHERE scope_key=? AND retired_at IS NULL ORDER BY key_id", (store.scope_key,))]


def setup(env, cited_relation):
    api, ops, store, _, _ = env
    s = message(env, "P2")
    p1_only = message(env, "P1")
    publish(env, "person:mom", s)
    uma = ops.create_entity("Uma")
    ops.add_alias(uma, "Uma", source_store_ids=[s])
    martha = ops.create_entity("Martha")
    ops.add_alias(martha, "Martha", source_store_ids=[s])
    rel_sources = [p1_only] if cited_relation else []
    ops.add_alias(martha, "my mom", "relation_phrase", owner_entity_id=uma, source_store_ids=rel_sources)
    jones = ops.create_entity("Mom Jones")
    ops.add_alias(jones, "Mom", "nickname", source_store_ids=[s])
    api.rebuild_entity_keys(store, sender_lookup=lambda sid: (uma, True, True) if sid == s else None)
    return s, uma, martha, jones


@pytest.mark.parametrize("cited_relation", [True, False], ids=["A2c-cited-P1-only", "A2-uncited"])
def test_owner_scoped_principal_read_matches_stored_key(env, cited_relation):
    api, ops, store, _, _ = env
    s, uma, martha, jones = setup(env, cited_relation)
    # Stored derivation: one owner-scoped key -> Martha (trusted relation path).
    assert active(store) == [("person:mom", uma, martha, "alias_match")]
    # Control: trusted read agrees with the stored key.
    assert api.resolve_key(store, "person:mom", s, owner_entity_id=uma) == martha
    # P2's own view exposes the stored key pointing at Martha.
    view = api.resolved_view(store, "P2")
    # r1b adaptation: item 7 makes P2's key view drop a binding whose only support is hidden.
    assert [(k["subject_key"], k["owner_entity_id"], k["entity_id"]) for k in view.get("keys", [])] in (
        [], [("person:mom", uma, martha)])
    # P2 cannot see the relation alias (P1-cited or uncited/owner-only).
    assert "relations" not in view
    got = api.resolve_key(store, "person:mom", s, owner_entity_id=uma, access_scope="P2")
    names = {uma: "Uma", martha: "Martha", jones: "Mom Jones", None: None}
    print("P2 owner-scoped resolve_key ->", names.get(got, got))
    assert got in (martha, None), f"returned {names.get(got, got)!r}"


def test_owner_principal_sees_uncited_relation(env):
    api, ops, store, _, _ = env
    s, uma, martha, jones = setup(env, cited_relation=False)
    got = api.resolve_key(store, "person:mom", s, owner_entity_id=uma, access_scope="P2", principal_is_owner=True)
    assert got == martha


def test_root_cause_control_fixed_key_candidates(env, monkeypatch):
    api, ops, store, _, _ = env
    s, uma, martha, jones = setup(env, cited_relation=True)
    original = api._key_candidates

    def fixed(view, subject_key, owner=None):
        namespace, _, value = subject_key.partition(":")
        if owner is not None and namespace not in {"user", "assistant"} and not api._relation_groups(view, value):
            return []
        return original(view, subject_key, owner)

    monkeypatch.setattr(api, "_key_candidates", fixed)
    assert api.resolve_key(store, "person:mom", s, owner_entity_id=uma, access_scope="P2") is None
    assert api.resolve_key(store, "person:mom", s, owner_entity_id=uma) == martha


def test_non_owner_name_path_unaffected_control(env):
    # Owner-less name key still resolves via nickname for P2 (sanity: the name path is live).
    api, ops, store, _, _ = env
    s, uma, martha, jones = setup(env, cited_relation=True)
    s2 = message(env, "P2")
    publish(env, "person:martha", s2)
    api.rebuild_entity_keys(store)
    assert api.resolve_key(store, "person:martha", s2, access_scope="P2") == martha
