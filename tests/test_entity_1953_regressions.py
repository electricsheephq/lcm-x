"""Permanent RED-first reproductions for #1953's nine bounded fixes."""

import json
import threading

import pytest

from tests.test_entity_bridge import active, bridge as bridge, publish, rebuild
from tests.test_entity_ops_identity import assert_no_production_imports


@pytest.mark.parametrize("scoped", [False, True])
def test_item1_hidden_owner_cannot_resolve_override(bridge, scoped):
    api, ops, store, _, _ = bridge
    owner, target = ops.create_entity("Owner"), ops.create_entity("Target")
    source = publish(bridge, "user:self")
    hidden = publish(bridge, "person:other", scope="P2")
    ops.add_alias(owner, "owner", source_store_ids=[hidden])
    ops.add_alias(target, "target", source_store_ids=[source])
    def lookup(sid):
        return owner, True, True
    rebuild(bridge, sender_lookup=lookup)
    ops.override_key("user:self", target, owner_entity_id=owner,
                     scope_store_ids=[source] if scoped else None)
    rebuild(bridge, sender_lookup=lookup)
    assert api.resolve_key(store, "user:self", source, owner_entity_id=owner) == target
    assert owner not in api.resolved_view(store, "P1")["canonical"]
    assert api.resolve_key(store, "user:self", source, owner_entity_id=owner,
                           access_scope="P1") is None


@pytest.mark.parametrize("namespace,kind", [("user", "person"), ("assistant", "agent")])
def test_item2_merged_profile_id_survives_rebuild(bridge, namespace, kind):
    api, ops, store, _, _ = bridge
    original, target = ops.create_entity("Original", kind), ops.create_entity("Target", kind)
    key = namespace + ":self"
    source = publish(bridge, key)
    def profiles(bound):
        return (original, None) if namespace == "user" else (None, original)
    rebuild(bridge, profile_entities=profiles)
    assert api.resolve_key(store, key, source) == original
    ops.merge(target, original)
    assert api.resolve_key(store, key, source) == target
    rebuild(bridge, profile_entities=profiles)
    assert api.resolve_key(store, key, source) == target


@pytest.mark.parametrize("spellings", [("sister", "my sister"), ("my sister", "sister")])
def test_item3_relation_spellings_merge_per_owner(bridge, spellings):
    api, ops, store, _, _ = bridge
    owners = [ops.create_entity("Owner") for _ in range(2)]
    targets = [ops.create_entity("Sister") for _ in range(2)]
    sources = [publish(bridge, "person:sister") for _ in range(2)]
    for owner, target, text in zip(owners, targets, spellings):
        ops.add_alias(target, text, "relation_phrase", owner_entity_id=owner)
    rebuild(bridge, sender_lookup=dict(zip(sources, [(o, True, True) for o in owners])).get)
    assert {(r["owner_entity_id"], r["entity_id"]) for r in active(store)} == set(zip(owners, targets))
    for owner, target, source in zip(owners, targets, sources):
        assert api.resolve_key(store, "person:sister", source, owner_entity_id=owner) == target
    # Both spellings for one owner must union candidates, never pick a winner.
    ops.add_alias(targets[1], "my sister", "relation_phrase", owner_entity_id=owners[0])
    assert api.resolve_key(store, "person:sister", sources[0], owner_entity_id=owners[0]) is None


def test_item4_add_key_checks_transaction_under_lock(bridge):
    _, ops, store, _, _ = bridge
    entity = ops.create_entity("Entity")
    attempted, finished = threading.Event(), threading.Event()
    errors, results = [], []

    def outsider():
        attempted.set()
        try:
            results.append(store.add_key("person:alex", entity, "alias_match", []))
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            finished.set()

    worker = threading.Thread(target=outsider)
    try:
        with pytest.raises(RuntimeError, match="rollback holder"):
            with store.transaction():
                worker.start()
                assert attempted.wait(2)
                # Broken add_key finishes inside our transaction; fixed add_key blocks.
                finished.wait(0.1)
                raise RuntimeError("rollback holder")
    finally:
        worker.join(3)
    assert not worker.is_alive()
    assert results == []
    assert errors == ["derived keys require transaction()"]
    assert active(store) == []


def test_item5_principal_view_does_not_rebind_stored_key(bridge):
    api, ops, store, _, _ = bridge
    source = publish(bridge, "person:alex", scope="P2")
    hidden = publish(bridge, "person:other")
    original = ops.create_entity("Alex")
    ops.add_alias(original, "alex", source_store_ids=[hidden])
    ops.add_alias(original, "visible identity", source_store_ids=[source])
    rebuild(bridge)
    assert active(store)[0]["entity_id"] == original
    other = ops.create_entity("Other Alex")
    ops.add_alias(other, "alex", source_store_ids=[source])
    assert api.resolve_key(store, "person:alex", source) is None
    assert api.resolved_view(store, "P2").get("keys", []) == []
    rebuild(bridge)
    assert api.resolved_view(store, "P2").get("keys", []) == []


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_item6_override_replay_rechecks_current_kind(bridge, scoped, owned):
    api, ops, store, _, _ = bridge
    target = ops.create_entity("Chosen")
    owner = ops.create_entity("Owner") if owned else None
    key = "person:sister" if owned else "person:alex"
    source = publish(bridge, key)
    if owned:
        ops.add_alias(target, "sister", "relation_phrase", owner_entity_id=owner)
    def lookup(sid):
        return (owner, True, True) if owned else None
    rebuild(bridge, sender_lookup=lookup)
    ops.override_key(key, target, owner_entity_id=owner,
                     scope_store_ids=[source] if scoped else None)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) == target
    change = ops.change_kind(target, "org")
    assert api.resolve_key(store, key, source, owner_entity_id=owner) is None
    rebuild(bridge, sender_lookup=lookup)
    assert not any(r["method"] == "override" for r in active(store))
    assert api.resolve_key(store, key, source, owner_entity_id=owner) is None
    ops.undo(change)
    rebuild(bridge, sender_lookup=lookup)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) == target


@pytest.mark.parametrize("scoped", [False, True])
def test_item6_incompatible_latest_override_does_not_revive_older(bridge, scoped):
    api, ops, store, _, _ = bridge
    older, newer = ops.create_entity("Older choice"), ops.create_entity("Newer choice")
    source = publish(bridge, "person:alex")
    options = dict(scope_store_ids=[source]) if scoped else {}
    ops.override_key("person:alex", older, **options)
    ops.override_key("person:alex", newer, **options)
    change = ops.change_kind(newer, "org")
    assert api.resolve_key(store, "person:alex", source) is None
    rebuild(bridge)
    assert not any(r["method"] == "override" for r in active(store))
    assert api.resolve_key(store, "person:alex", source) is None
    ops.undo(change)
    rebuild(bridge)
    assert api.resolve_key(store, "person:alex", source) == newer


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("scopes", [(True, False), (False, True), (False, False), (True, True)])
def test_item6_latest_incompatible_override_uses_alias_fallback(bridge, owned, scopes):
    api, ops, store, _, _ = bridge
    original, older, newer = [ops.create_entity(name) for name in
                              ("Alias target", "Older choice", "Newer choice")]
    owner = ops.create_entity("Owner") if owned else None
    key = "person:sister" if owned else "person:alex"
    source = publish(bridge, key)
    if owned:
        ops.add_alias(original, "sister", "relation_phrase", owner_entity_id=owner)
    else:
        ops.add_alias(original, "alex")

    def lookup(sid):
        return (owner, True, True) if owned else None

    rebuild(bridge, sender_lookup=lookup)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) == original
    for target, scoped in zip((older, newer), scopes):
        ops.override_key(key, target, owner_entity_id=owner,
                         scope_store_ids=[source] if scoped else None)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) == newer
    change = ops.change_kind(newer, "org")
    before = api.resolve_key(store, key, source, owner_entity_id=owner)
    rebuild(bridge, sender_lookup=lookup)
    after = api.resolve_key(store, key, source, owner_entity_id=owner)
    # ADR D6: without a compatible override, the unique compatible alias answers.
    assert (before, after) == (original, original)
    ops.undo(change)
    rebuild(bridge, sender_lookup=lookup)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) == newer


@pytest.mark.parametrize("scoped", [False, True])
def test_item7_relation_requires_owner_before_override_replay(bridge, scoped):
    api, ops, store, _, _ = bridge
    named, target = ops.create_entity("Sister"), ops.create_entity("Chosen")
    ops.add_alias(named, "sister")
    source = publish(bridge, "person:sister")
    rebuild(bridge)
    ops.override_key("person:sister", target, scope_store_ids=[source] if scoped else None)
    assert api.resolve_key(store, "person:sister", source) == target
    owner, sister = ops.create_entity("Owner"), ops.create_entity("Relation")
    ops.add_alias(sister, "sister", "relation_phrase", owner_entity_id=owner)
    assert api.resolve_key(store, "person:sister", source) is None
    rebuild(bridge, sender_lookup=lambda sid: (owner, False, True))
    assert api.resolve_key(store, "person:sister", source) is None
    assert not any(r["owner_entity_id"] is None for r in active(store))


def test_item8_owner_fallback_checks_complete_citation_visibility(bridge):
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("Owner")
    visible = publish(bridge, "user:self")
    hidden = publish(bridge, "user:self", scope="P2")
    ops.add_alias(owner, "owner", source_store_ids=[visible])
    rebuild(bridge, sender_lookup=lambda sid: (owner, True, True))
    assert set(json.loads(active(store)[0]["source_store_ids"])) == {visible, hidden}
    assert api.resolve_key(store, "user:self", visible, owner_entity_id=owner) == owner
    assert api.resolved_view(store, "P1").get("keys", []) == []
    assert api.resolve_key(store, "user:self", visible, owner_entity_id=owner,
                           access_scope="P1") is None
    assert api.resolve_key(store, "user:self", visible, owner_entity_id=owner,
                           access_scope="P1", principal_is_owner=True) is None


@pytest.mark.parametrize("directory", [".venv", "venv", "build", "dist", "site-packages", "node_modules"])
def test_item9_import_guard_ignores_generated_trees(tmp_path, directory):
    module = tmp_path / directory / "nested" / "legacy.py"
    module.parent.mkdir(parents=True)
    module.write_text("print 'python two'\n")
    assert_no_production_imports(tmp_path)


@pytest.mark.parametrize("restore", ["person", "other", "topic", "undo"])
@pytest.mark.parametrize("shape", ["owned_global", "owned_scoped_then_global", "owned_global_global",
                                   "owned_scoped", "unowned_global"])
def test_item6_recompatible_override_same_sender_rebuild(bridge, shape, restore):
    """Permanent reviewer recompat shapes: restoration is checked BEFORE rebuild."""
    api, ops, store, _, _ = bridge
    original, older, newer = [ops.create_entity(name) for name in ("S", "A", "B")]
    owned = shape.startswith("owned")
    owner = ops.create_entity("Owner") if owned else None
    key = "person:sister" if owned else "person:alex"
    source = publish(bridge, key)
    if owned:
        ops.add_alias(original, "sister", "relation_phrase", owner_entity_id=owner)
    else:
        ops.add_alias(original, "alex")
    def lookup(sid):
        return (owner, True, True) if owned else None
    rebuild(bridge, sender_lookup=lookup)
    if shape in {"owned_scoped_then_global", "owned_global_global"}:
        ops.override_key(key, older, owner_entity_id=owner,
                         scope_store_ids=[source] if shape == "owned_scoped_then_global" else None)
    ops.override_key(key, newer, owner_entity_id=owner,
                     scope_store_ids=[source] if shape == "owned_scoped" else None)
    change = ops.change_kind(newer, "org")
    rebuild(bridge, sender_lookup=lookup)
    assert api.resolve_key(store, key, source, owner_entity_id=owner) == original
    if restore == "undo":
        ops.undo(change)
    else:
        ops.change_kind(newer, restore)
    before = api.resolve_key(store, key, source, owner_entity_id=owner)
    rebuild(bridge, sender_lookup=lookup)
    after = api.resolve_key(store, key, source, owner_entity_id=owner)
    rebuild(bridge, sender_lookup=lookup)
    assert (before, after, api.resolve_key(store, key, source, owner_entity_id=owner)) == (newer,) * 3


@pytest.mark.parametrize("seed,member,restore", [(2169, 1, "topic"), (1951, 2, "other")])
def test_item6_minimized_recompatible_seed(bridge, seed, member, restore):
    """Port the reviewer's minimized eight-op traces, not its scratch generator."""
    api, ops, store, _, _ = bridge
    owner = ops.create_entity("O1")
    targets = [ops.create_entity(f"T{i}") for i in range(3)]
    source = publish(bridge, "person:sister")
    ops.add_alias(targets[0], "sister", "relation_phrase", owner_entity_id=owner)
    def lookup(sid):
        return owner, True, True
    rebuild(bridge, sender_lookup=lookup)
    ops.override_key("person:sister", targets[member], owner_entity_id=owner)
    ops.change_kind(targets[member], "org")
    rebuild(bridge, sender_lookup=lookup)
    ops.change_kind(targets[member], restore)
    before = api.resolve_key(store, "person:sister", source, owner_entity_id=owner)
    rebuild(bridge, sender_lookup=lookup)
    after = api.resolve_key(store, "person:sister", source, owner_entity_id=owner)
    assert (before, after) == (targets[member], targets[member]), seed
