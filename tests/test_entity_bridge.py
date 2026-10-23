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
    for key in ("user:self", "assistant:self"):
        publish(bridge, key)
    rebuild(bridge)
    assert active(store) == []
    rebuild(bridge, profile_entities=lambda bound: (user, agent))
    assert {(r["subject_key"], r["entity_id"]) for r in active(store)} == {
        ("user:self", user), ("assistant:self", agent)}


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
