"""W1b acceptance: derived keys, source binding, scoped overrides and undo."""

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


def publish(bridge, key, source=None, version="assertions-v1", scope="P1"):
    _, _, _, messages, assertions = bridge
    if source is None:
        source = messages.append("S", {"role": "user", "content": "synthetic"}, source="cli")
        messages.connection.execute("UPDATE messages SET access_scope=? WHERE store_id=?", (scope, source))
        messages.connection.commit()
    candidate = AssertionCandidate(source_span_start=0, source_span_end=9, subject_key=key,
        predicate_key="likes", object_value="tea", value_text="tea", kind="fact")
    assertions.publish_source(assertions.snapshot_source(source), [candidate], extraction_version=version)
    return source


def active(store):
    return [dict(r) for r in store.connection.execute(
        "SELECT * FROM lcm_entity_keys WHERE scope_key=? AND retired_at IS NULL ORDER BY key_id", (store.scope_key,))]


def rebuild(bridge, **options):
    api, _, store, _, _ = bridge
    return api.rebuild_entity_keys(store, **options)


@pytest.mark.parametrize("namespace,kind", [("person", "person"), ("organization", "org"),
    ("project", "project"), ("place", "place"), ("object", "thing"), ("account", "thing"),
    ("document", "thing"), ("event", None), ("conversation", None), ("future", None)])
def test_namespace_table_resolved_kinds_and_no_rebuild_events(bridge, namespace, kind):
    api, ops, store, _, _ = bridge
    entity = ops.create_entity("Alex", "other")
    ops.add_alias(entity, "Alex")
    if kind:
        ops.change_kind(entity, kind)
    publish(bridge, namespace + ":alex")
    before = store.events_after()
    rebuild(bridge)
    assert [r["entity_id"] for r in active(store)] == ([entity] if kind else [])
    assert store.events_after() == before
    changes = store.connection.total_changes
    assert rebuild(bridge) == 0
    assert store.connection.total_changes == changes
    assert api.replay_projection(store)["entities"][entity] == ("active", None)


@pytest.mark.parametrize("kind", ["topic", "other", "org"])
def test_incompatible_ambiguous_and_absent_family(bridge, kind):
    _, ops, store, _, _ = bridge
    for _ in range(2):
        entity = ops.create_entity("Alex", kind)
        ops.add_alias(entity, "alex")
    publish(bridge, "person:alex")
    rebuild(bridge)
    assert active(store) == []


def test_absent_and_invalid_assertion_family(tmp_path, bridge):
    api = bridge[0]
    path = tmp_path / "absent.db"
    with closing(MessageStore(path)), closing(EntityStore(path, profile_key="A")) as store:
        assert api.rebuild_entity_keys(store) == 0
    entity = bridge[1].create_entity("Alex")
    bridge[1].add_alias(entity, "Alex")
    publish(bridge, "person:alex")
    bridge[2].connection.execute("DROP INDEX idx_lcm_assertions_source")
    with pytest.raises(api.EntityIntegrityError):
        rebuild(bridge)
    assert active(bridge[2]) == []


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_owner_scoped_senders_and_rejected_speech(bridge, namespace):
    _, ops, store, _, _ = bridge
    owners = [ops.create_entity("speaker") for _ in range(2)]
    targets = owners if namespace == "user" else [ops.create_entity("sister") for _ in range(2)]
    for owner, target in zip(owners, targets):
        ops.add_alias(target, "sister", "relation_phrase", owner_entity_id=owner)
    key = namespace + (":self" if namespace == "user" else ":sister")
    sources = [publish(bridge, key) for _ in range(2)]
    rebuild(bridge)
    assert active(store) == []
    lookup = dict(zip(sources, [(o, True, True) for o in owners])).get
    rebuild(bridge, sender_lookup=lookup)
    rows = active(store)
    assert {(r["owner_entity_id"], r["entity_id"], tuple(json.loads(r["source_store_ids"]))) for r in rows} == {
        (o, t, (s,)) for o, t, s in zip(owners, targets, sources)}
    for verified, first_person in [(False, True), (True, False), (False, False)]:
        rebuild(bridge, sender_lookup=lambda source: (owners[0], verified, first_person))
        assert active(store) == []


def test_profile_user_assistant_defaults_and_injected_entities(bridge):
    _, ops, store, _, _ = bridge
    user, agent = ops.create_entity("User"), ops.create_entity("Agent", "agent")
    ops.add_alias(user, "self")
    ops.add_alias(agent, "self")
    sources = []
    for key in ("user:self", "assistant:self"):
        sources.append(publish(bridge, key))
    rebuild(bridge)
    assert active(store) == []
    for key, source in zip(("user:self", "assistant:self"), sources):
        assert bridge[0].resolve_key(store, key, source) is None
    rebuild(bridge, profile_entities=lambda bound: (user, agent))
    assert {(r["subject_key"], r["entity_id"]) for r in active(store)} == {
        ("user:self", user), ("assistant:self", agent)}


def test_bridge_platform_matching_preserves_identifier_bytes(bridge):
    api, ops, store, _, _ = bridge
    entity = ops.create_entity("platform sender")
    ops.add_alias(entity, "abc", "platform_id", platform="tg", verified=True)
    mismatch = publish(bridge, "person:tg:ABC")
    exact = publish(bridge, "person:tg:abc")
    rebuild(bridge)
    assert api.resolve_key(store, "person:tg:ABC", mismatch) is None
    assert api.resolve_key(store, "person:tg:abc", exact) == entity
    assert [r["subject_key"] for r in active(store)] == ["person:tg:abc"]


@pytest.mark.parametrize("scoped", [False, True])
def test_override_source_binding_churn_undo_and_replay(bridge, scoped, monkeypatch):
    api, ops, store, _, assertions = bridge
    original, chosen = ops.create_entity("Alex"), ops.create_entity("Chosen", "other")
    ops.add_alias(original, "Alex")
    source = publish(bridge, "person:alex")
    rebuild(bridge)
    before = api.resolved_view(store)
    event_count = len(store.events_after())
    event = ops.override_key("person:alex", chosen, scope_store_ids=[source] if scoped else None)
    assert len(store.events_after()) == event_count + 1
    p = json.loads(store.events_after()[-1]["payload_json"])
    assert set(p) >= {"created", "changes", "touches"}
    assert p["touches"]["keys"] == [["person:alex", None]]
    assert bool(p["created"].get("keys")) is not scoped
    assert api.resolve_key(store, "person:alex", source) == chosen
    new_source = publish(bridge, "person:alex")
    rebuild(bridge)
    assert api.resolve_key(store, "person:alex", source) == chosen
    assert api.resolve_key(store, "person:alex", new_source) == original
    # The same source under a new extraction version remains bound.
    publish(bridge, "person:alex", source, version="assertions-v2")
    monkeypatch.setattr(importlib.import_module("hermes_lcm.assertion_store"), "CURRENT_EXTRACTION_VERSION", "assertions-v2")
    rebuild(bridge)
    assert api.resolve_key(store, "person:alex", source) == chosen
    publish(bridge, "person:renamed", source, version="assertions-v3")
    monkeypatch.setattr(importlib.import_module("hermes_lcm.assertion_store"), "CURRENT_EXTRACTION_VERSION", "assertions-v3")
    rebuild(bridge)
    assert api.resolve_key(store, "person:renamed", source) is None
    ops.add_alias(original, "renamed")
    renamed_event = store.events_after()[-1]["event_id"]
    rebuild(bridge)
    assert api.resolve_key(store, "person:renamed", source) == original
    assertions.invalidate_source(source, reason="test")
    rebuild(bridge)
    assert active(store) == []
    assert api.resolve_key(store, "person:alex", source) is None
    assert api.replay_projection(store)["entities"][chosen] == ("active", None)
    # New evidence with reused text never wakes the old override.
    unrelated = publish(bridge, "person:alex", version="assertions-v3")
    rebuild(bridge)
    assert api.resolve_key(store, "person:alex", unrelated) == original
    assert before["cards"][original]["display_name"] == "Alex"
    ops.undo(renamed_event)
    # Restoration collisions fail with a typed refusal, never IntegrityError.
    if not scoped:
        with pytest.raises(api.EntityOpRefused, match="key_held"):
            ops.undo(event)
    assertions.invalidate_source(unrelated, reason="test")
    rebuild(bridge)
    ops.undo(event)
    rebuild(bridge)
    assert api.resolve_key(store, "person:alex", unrelated) is None


def test_override_ambiguity_scoped_storage_dependency_and_inverse(bridge):
    api, ops, store, _, _ = bridge
    a, b = ops.create_entity("A"), ops.create_entity("B")
    aa = ops.add_alias(a, "alex")
    ops.add_alias(b, "alex")
    source = publish(bridge, "person:alex")
    rebuild(bridge)
    with pytest.raises(api.EntityOpRefused, match="ambiguous_key"):
        ops.override_key("person:alex", a)
    event = ops.override_key("person:alex", a, scope_store_ids=[source])
    assert active(store) == []
    newer = ops.override_key("person:alex", b, scope_store_ids=[source])
    with pytest.raises(api.EntityOpRefused, match="dependent_events") as error:
        ops.undo(event)
    assert error.value.blocking_event_ids == [newer]
    ops.undo(newer)
    assert api.resolve_key(store, "person:alex", source) == a
    ops.undo(event)
    assert api.resolve_key(store, "person:alex", source) is None
    ops.retire_alias(aa)
    rebuild(bridge)
    before = api.resolved_view(store)
    event = ops.override_key("person:alex", a)
    assert [r["method"] for r in active(store)] == ["override"]
    ops.undo(event)
    assert [r["method"] for r in active(store)] == ["alias_match"]
    assert api.resolved_view(store) == before


@pytest.mark.parametrize("operation", ["merge", "split"])
def test_identity_with_keys_and_d8(bridge, operation):
    api, ops, store, _, _ = bridge
    a, b = ops.create_entity("A"), ops.create_entity("B")
    source = publish(bridge, "person:alex")
    alias = ops.add_alias(a, "Alex", source_store_ids=[source])
    rebuild(bridge)
    before = api.resolved_view(store)
    assert before["keys"][0]["entity_id"] == a
    assert "person:alex" not in json.dumps(api.resolved_view(store, "P2"))
    assert api.resolved_view(store, "").get("keys", []) == []
    if operation == "merge":
        ops.merge(b, a)
    else:
        ops.split(a, [alias], [], "new")
    event = store.events_after()[-1]["event_id"]
    rebuild(bridge)
    ops.undo(event)
    rebuild(bridge)
    assert api.resolved_view(store) == before


def test_current_sources_only_scope_override_and_read_cost(bridge):
    api, ops, store, messages, assertions = bridge
    a, b = ops.create_entity("A"), ops.create_entity("B")
    ops.add_alias(a, "alex")
    current = publish(bridge, "person:alex")
    stale = publish(bridge, "person:alex")
    deleted = publish(bridge, "person:alex")
    publish(bridge, "person:old", version="old")
    assertions.invalidate_source(stale, reason="test")
    messages.connection.execute("DELETE FROM messages WHERE store_id=?", (deleted,))
    messages.connection.commit()
    rebuild(bridge)
    assert json.loads(active(store)[0]["source_store_ids"]) == [current]
    trace = []
    store.connection.set_trace_callback(trace.append)
    assert api.resolve_key(store, "person:alex", current) == a
    baseline = len(trace)
    store.connection.set_trace_callback(None)
    ops.override_key("person:alex", b, scope_store_ids=[current])
    trace.clear()
    store.connection.set_trace_callback(trace.append)
    assert api.resolve_key(store, "person:alex", current) == b
    store.connection.set_trace_callback(None)
    assert len(trace) <= baseline + 1
    with closing(EntityStore(store.db_path, profile_key="B")) as other:
        bound = api.EntityOps(other)
        before = list(other.connection.iterdump())
        for foreign in (a, "ent_missing"):
            with pytest.raises(api.EntityOpRefused, match="not_found"):
                bound.override_key("person:alex", foreign, scope_store_ids=[current])
        assert list(other.connection.iterdump()) == before
        assert api.resolve_key(other, "person:alex", current) is None


def test_explicit_owner_overrides_never_coalesce_and_foreign_owner_refused(bridge):
    api, ops, store, _, _ = bridge
    owners = [ops.create_entity("owner") for _ in range(2)]
    sisters = [ops.create_entity("sister") for _ in range(2)]
    for owner, sister in zip(owners, sisters):
        ops.add_alias(sister, "my sister", "relation_phrase", owner_entity_id=owner)
    sources = [publish(bridge, "person:sister") for _ in range(2)]
    with pytest.raises(api.EntityOpRefused, match="missing_owner"):
        ops.override_key("person:sister", sisters[0], scope_store_ids=[sources[0]])
    for owner, sister, source in zip(owners, sisters, sources):
        event = ops.override_key("person:sister", sister, scope_store_ids=[source], owner_entity_id=owner)
        assert json.loads(store.events_after()[-1]["payload_json"])["touches"]["keys"] == [["person:sister", owner]]
        assert api.resolve_key(store, "person:sister", source, owner_entity_id=owner) == sister
        assert api.resolve_key(store, "person:sister", source, owner_entity_id=owners[1 - owners.index(owner)]) is None
        assert store.events_after()[-1]["event_id"] == event
    with closing(EntityStore(store.db_path, profile_key="B")) as other:
        bound = api.EntityOps(other)
        entity = bound.create_entity("local")
        before = list(other.connection.iterdump())
        for foreign in (owners[0], "ent_missing"):
            with pytest.raises(api.EntityOpRefused, match="not_found"):
                bound.override_key("person:sister", entity, scope_store_ids=sources, owner_entity_id=foreign)
            with pytest.raises(api.EntityOpRefused, match="not_found"):
                api.resolve_key(other, "person:sister", sources[0], owner_entity_id=foreign)
        assert list(other.connection.iterdump()) == before


@pytest.mark.parametrize("scoped", [False, True])
def test_key_source_visibility_and_scoped_override_view_cost(bridge, scoped):
    api, ops, store, _, _ = bridge
    a, b = ops.create_entity("A"), ops.create_entity("B", "other")
    visible = publish(bridge, "person:alex", scope="P1")
    hidden = publish(bridge, "person:alex", scope="P2")
    ops.add_alias(a, "alex", source_store_ids=[visible])
    ops.add_alias(b, "chosen", source_store_ids=[visible])
    rebuild(bridge)
    assert api.resolve_key(store, "person:alex", hidden, access_scope="P1") is None
    ops.override_key("person:alex", b, scope_store_ids=[visible, hidden] if scoped else None)
    assert api.resolve_key(store, "person:alex", hidden, access_scope="P1") is None
    trace = []
    store.connection.set_trace_callback(trace.append)
    api.resolved_view(store, "P1")
    store.connection.set_trace_callback(None)
    assert sum(sql.lstrip().upper().startswith("SELECT") for sql in trace) == 5


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("namespace,kind,accepted", [
    ("event", "person", False), ("conversation", "other", False),
    ("future", "topic", False), ("person", "org", False),
    ("person", "topic", True), ("person", "other", True)])
def test_r1b_1_override_namespace_and_resolved_kind(bridge, scoped, namespace, kind, accepted):
    api, ops, store, _, _ = bridge
    target = ops.create_entity("Chosen")
    ops.change_kind(target, kind)
    subject = namespace + ":alex"
    source = publish(bridge, subject)
    before = list(store.connection.iterdump())
    options = dict(scope_store_ids=[source]) if scoped else {}
    if accepted:
        ops.override_key(subject, target, **options)
        assert api.resolve_key(store, subject, source) == target
    else:
        with pytest.raises(api.EntityOpRefused):
            ops.override_key(subject, target, **options)
        assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("owner_scoped", [False, True])
def test_r1b_2_hidden_relation_never_falls_through_to_name(bridge, owner_scoped):
    api, ops, store, _, _ = bridge
    source = publish(bridge, "person:mom", scope="P2")
    owner, mom, unrelated = [ops.create_entity(name) for name in ("Owner", "Martha", "Mom Jones")]
    for entity, alias in [(owner, "Owner"), (mom, "Martha"), (unrelated, "mom")]:
        ops.add_alias(entity, alias, source_store_ids=[source])
    ops.add_alias(mom, "my mom", "relation_phrase", owner_entity_id=owner)
    rebuild(bridge, sender_lookup=lambda sid: (owner, True, True))
    options = dict(owner_entity_id=owner) if owner_scoped else {}
    assert api.resolve_key(store, "person:mom", source, **options) == (mom if owner_scoped else None)
    assert api.resolve_key(store, "person:mom", source, access_scope="P2", **options) is None


@pytest.mark.parametrize("operation", ["rebuild", "override", "undo"])
def test_r1b_3_deleted_session_key_rows_never_collide(bridge, operation):
    api, ops, store, messages, assertions = bridge
    original, chosen = ops.create_entity("Original"), ops.create_entity("Chosen")
    ops.add_alias(original, "alex")
    source = publish(bridge, "person:alex")
    rebuild(bridge)
    if operation == "undo":
        event = ops.override_key("person:alex", chosen)
        assertions.invalidate_source(source, reason="test")
        publish(bridge, "person:alex")
        rebuild(bridge)
    assert messages.delete_session_messages("S") > 0
    if operation == "rebuild":
        rebuild(bridge)
        assert active(store) == []
        current = publish(bridge, "person:alex")
        rebuild(bridge)
        assert api.resolve_key(store, "person:alex", current) == original
    elif operation == "override":
        publish(bridge, "person:alex")
        ops.override_key("person:alex", chosen)
    else:
        with pytest.raises(api.EntityOpRefused, match="key_held"):
            ops.undo(event)


@pytest.mark.parametrize("churn", ["shrink", "merge"])
def test_r1b_4_global_override_rederived_row_is_undoable(bridge, churn):
    api, ops, store, _, assertions = bridge
    original, chosen, merged = [ops.create_entity(name) for name in ("Original", "Chosen", "Merged")]
    ops.add_alias(original, "alex")
    first, second = publish(bridge, "person:alex"), publish(bridge, "person:alex")
    rebuild(bridge)
    event = ops.override_key("person:alex", chosen)
    if churn == "shrink":
        assertions.invalidate_source(second, reason="test")
    else:
        ops.merge(merged, chosen)
        merge_event = store.events_after()[-1]["event_id"]
        rebuild(bridge)
        ops.undo(merge_event)
    rebuild(bridge)
    ops.undo(event)
    rebuild(bridge)
    assert api.resolve_key(store, "person:alex", first) == original
    assert {row["method"] for row in active(store)} == {"alias_match"}


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_r1b_5_global_override_cites_only_owners_sources(bridge, namespace):
    api, ops, store, _, _ = bridge
    owners = [ops.create_entity("Owner") for _ in range(2)]
    targets = owners if namespace == "user" else [ops.create_entity("Sister") for _ in range(2)]
    chosen = ops.create_entity("Chosen", "other")
    subject = namespace + (":self" if namespace == "user" else ":sister")
    sources = [publish(bridge, subject, scope=scope) for scope in ("P1", "P2")]
    for owner, target, source in zip(owners, targets, sources):
        ops.add_alias(owner, "owner", source_store_ids=[source])
        ops.add_alias(target, "my sister", "relation_phrase", owner_entity_id=owner, source_store_ids=[source])
    ops.add_alias(chosen, "chosen", source_store_ids=[sources[0]])
    lookup = dict(zip(sources, [(owner, True, True) for owner in owners])).get
    rebuild(bridge, sender_lookup=lookup)
    ops.override_key(subject, chosen, owner_entity_id=owners[0])
    payload = json.loads(store.events_after()[-1]["payload_json"])["override"]
    assert payload["scope_store_ids"] == [sources[0]]
    rebuild(bridge, sender_lookup=lookup)
    row = next(r for r in active(store) if r["owner_entity_id"] == owners[0])
    assert json.loads(row["source_store_ids"]) == [sources[0]]
    assert api.resolve_key(store, subject, sources[1], owner_entity_id=owners[0]) is None
    assert api.resolve_key(store, subject, sources[0], owner_entity_id=owners[0], access_scope="P1") == chosen


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_r1b_6_owner_override_leaves_new_evidence_to_aliases(bridge, namespace):
    api, ops, store, _, _ = bridge
    owner, sister, chosen = [ops.create_entity(name) for name in ("Owner", "Sister", "Chosen")]
    ops.add_alias(sister, "my sister", "relation_phrase", owner_entity_id=owner)
    subject = namespace + (":self" if namespace == "user" else ":sister")
    original = owner if namespace == "user" else sister
    first = publish(bridge, subject)
    def lookup(sid):
        return owner, True, True
    rebuild(bridge, sender_lookup=lookup)
    ops.override_key(subject, chosen, owner_entity_id=owner)
    new = publish(bridge, subject)
    rebuild(bridge, sender_lookup=lookup)
    assert api.resolve_key(store, subject, first, owner_entity_id=owner) == chosen
    assert api.resolve_key(store, subject, new, owner_entity_id=owner) == original
    assert json.loads(active(store)[0]["source_store_ids"]) == [first]
    assert rebuild(bridge, sender_lookup=lookup) == 0


def test_r1b_7_principal_keys_require_visible_alias_support(bridge):
    api, ops, store, _, _ = bridge
    source = publish(bridge, "person:alex", scope="P2")
    hidden = publish(bridge, "person:other", scope="P1")
    target = ops.create_entity("Alex")
    ops.add_alias(target, "alex", source_store_ids=[hidden])
    ops.add_alias(target, "visible identity", source_store_ids=[source])
    rebuild(bridge)
    assert api.resolved_view(store)["keys"]
    assert api.resolve_key(store, "person:alex", source, access_scope="P2") is None
    assert api.resolved_view(store, "P2").get("keys", []) == []
    ops.add_alias(target, "alex", source_store_ids=[source])
    assert api.resolved_view(store, "P2")["keys"][0]["entity_id"] == target


def test_r1b_m10_read_candidates_use_resolved_compatible_kind(bridge):
    api, ops, store, _, _ = bridge
    target = ops.create_entity("Alex", "person")
    ops.add_alias(target, "alex")
    source = publish(bridge, "person:alex")
    rebuild(bridge)
    ops.change_kind(target, "org")
    assert api.resolve_key(store, "person:alex", source) is None


def test_r1b_m1_profile_entities_require_compatible_kind(bridge):
    _, ops, store, _, _ = bridge
    wrong_user, wrong_agent = ops.create_entity("User", "agent"), ops.create_entity("Agent", "person")
    publish(bridge, "user:self")
    publish(bridge, "assistant:self")
    rebuild(bridge, profile_entities=lambda bound: (wrong_user, wrong_agent))
    assert active(store) == []


@pytest.mark.parametrize("namespace,kind", [("user", "person"), ("assistant", "agent")])
def test_r1b_m2_profile_names_never_supply_automatic_candidates(bridge, namespace, kind):
    api, ops, store, _, _ = bridge
    target = ops.create_entity("Self", kind)
    ops.add_alias(target, "self")
    source = publish(bridge, namespace + ":self")
    assert api._key_candidates(api.resolved_view(store), namespace + ":self") == []
    assert api.resolve_key(store, namespace + ":self", source) is None
