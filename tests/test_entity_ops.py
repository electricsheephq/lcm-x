"""W1a: event projection, inverse operations and scoped resolved reads."""

from contextlib import closing
import importlib
import json
import random
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_lcm.entity_store import EntityStore
from hermes_lcm.store import MessageStore


@pytest.fixture
def api():
    return importlib.import_module("hermes_lcm.entity_ops")


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "lcm.db"
    with closing(MessageStore(path)) as messages:
        messages.connection.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
        for key, scope in [(1, "P1"), (2, "P2")]:
            messages.connection.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp,access_scope) "
                                        "VALUES(?,'S','user','synthetic',1,?)", (key, scope))
        messages.connection.commit()
    with closing(EntityStore(path, profile_key="profile-A")) as result:
        yield result


@pytest.fixture
def ops(store, api):
    return api.EntityOps(store)


def last(store):
    return store.events_after()[-1]["event_id"]


def payload(store):
    return json.loads(store.events_after()[-1]["payload_json"])


def projection(store):
    result = {}
    for table, key, fields in [("entities", "entity_id", "status,merged_into"),
                               ("aliases", "alias_id", "retired_at"),
                               ("notes", "note_id", "superseded_by")]:
        name = {"entities": "lcm_entities", "aliases": "lcm_entity_aliases", "notes": "lcm_memory_notes"}[table]
        result[table] = {row[0]: tuple(row)[1:] for row in store.connection.execute(
            f"SELECT {key},{fields} FROM {name} WHERE scope_key=?", (store.scope_key,))}
    return result


def scope_rows(store):
    return {table: [tuple(r) for r in store.connection.execute(
        f"SELECT * FROM {table} WHERE scope_key=? ORDER BY rowid", (store.scope_key,))]
        for table in ("lcm_entities", "lcm_entity_aliases", "lcm_memory_notes", "lcm_memory_events")}


def refused(api, reason, call):
    with pytest.raises(api.EntityOpRefused) as caught:
        call()
    assert caught.value.reason == reason
    return caught.value


def test_r1_platform_fallback_keeps_identifier_bytes(store, ops, api):
    entity = ops.create_entity("pid")
    ops.add_alias(entity, "abc", "platform_id", platform="tg", verified=True)
    assert ops.lookup_alias("tg:ABC") == {"candidates": [], "ambiguous": False}
    assert ops.holder_for_platform_id("tg:ABC") is None
    name = ops.create_entity("colon name")
    ops.add_alias(name, "TG:ABC")
    assert ops.lookup_alias("tg:ABC") == {"candidates": [name], "ambiguous": False}
    assert ops.lookup_alias("tg:abc") == {"candidates": sorted([name, entity]), "ambiguous": True}


@pytest.mark.parametrize("pid", ["abc", "tg:", ":x"])
@pytest.mark.parametrize("operation", ["holder", "attach"])
@pytest.mark.parametrize("verified", [False, True])
def test_r2_invalid_platform_ids_are_nonmutating(store, ops, api, pid, operation, verified):
    before = scope_rows(store)
    call = (lambda: ops.holder_for_platform_id(pid)) if operation == "holder" else (
        lambda: ops.attach_or_create_for_platform_id(pid, "invalid", verified=verified))
    refused(api, "invalid_platform_id", call)
    assert scope_rows(store) == before


@pytest.mark.parametrize("verified", [False, True])
def test_r2_add_platform_alias_requires_platform(store, ops, api, verified):
    entity = ops.create_entity("pid")
    before = scope_rows(store)
    refused(api, "invalid_platform_id", lambda: ops.add_alias(entity, "abc", "platform_id", verified=verified))
    assert scope_rows(store) == before


def test_r3_foreign_footprint_blocks_undo_and_replay_fails_loud(store, ops, api):
    entity = ops.create_entity("entity")
    target = last(store)
    store.add_alias(entity, "direct mutation")
    before = scope_rows(store)
    error = refused(api, "dependent_events", lambda: ops.undo(target))
    assert error.blocking_event_ids == [last(store)]
    assert scope_rows(store) == before
    with pytest.raises(api.EntityIntegrityError):
        api.replay_projection(store)


def test_r3_replay_rejects_store_direct_events(store, api):
    store.create_entity("direct source mutation")
    with pytest.raises(api.EntityIntegrityError):
        api.replay_projection(store)


def test_r5_relation_candidates_are_separate_by_owner(store, ops, api):
    owners = [ops.create_entity("owner") for _ in range(2)]
    sisters = [ops.create_entity("sister") for _ in range(2)]
    for owner, sister in zip(owners, sisters):
        ops.add_alias(sister, "my sister", "relation_phrase", owner_entity_id=owner)
    view = api.resolved_view(store)
    assert "my sister" not in view["aliases"]
    assert view["relations"]["my sister"] == {
        owner: {"candidates": [sister], "ambiguous": False} for owner, sister in zip(owners, sisters)}


def test_merge_round_trip_kind_and_canonical_chain(store, ops, api):
    a, b, c = [ops.create_entity(name) for name in "ABC"]
    original = api.resolved_view(store)
    ops.merge(a, b)
    first = last(store)
    ops.merge(b, c)
    assert api.resolved_view(store)["canonical"][c] == a
    refused(api, "same_canonical", lambda: ops.merge(c, b))
    refused(api, "same_canonical", lambda: ops.merge(a, a))
    ops.undo(last(store))
    ops.undo(first)
    assert api.resolved_view(store) == original
    ops.change_kind(b, "org")
    assert store.get_entity(b)["kind"] == "person"
    refused(api, "kind_mismatch", lambda: ops.merge(a, b))
    ops.change_kind(a, "org")
    ops.merge(a, b)
    assert api.resolved_view(store)["cards"][a]["kind"] == "org"


def test_split_lifo_round_trip_and_statement_level_holder_probe(store, ops, api):
    entity = ops.create_entity("source")
    alias = ops.add_alias(entity, "AbＣ", "platform_id", platform="Telegram", verified=True)
    note = ops.add_note("fact", entity_id=entity, cites=[1])
    original = api.resolved_view(store)
    probes = []

    def probe(sql):
        if sql.lstrip().upper().startswith(("UPDATE", "INSERT")):
            counts = store.connection.execute("SELECT alias_norm,count(*) FROM lcm_entity_aliases "
                "WHERE alias_kind='platform_id' AND retired_at IS NULL GROUP BY scope_key,alias_norm").fetchall()
            probes.append([r[1] for r in counts])

    store.connection.set_trace_callback(probe)
    try:
        second = ops.split(entity, [alias], [note], "second")
        event1 = last(store)
        moved_alias = store.aliases_for_entity(second)[0]["alias_id"]
        moved_note = store.notes_for_entity(second)[0]["note_id"]
        third = ops.split(second, [moved_alias], [moved_note], "third")
        event2 = last(store)
        error = refused(api, "dependent_events", lambda: ops.undo(event1))
        assert error.blocking_event_ids == [event2]
        assert api.resolved_view(store)["cards"][third]
        ops.undo(event2)
        ops.undo(event1)
    finally:
        store.connection.set_trace_callback(None)
    assert probes and all(count <= 1 for counts in probes for count in counts)
    assert api.resolved_view(store) == original
    assert api.replay_projection(store) == projection(store)
    assert store.get_alias(alias)["alias_norm"] == "telegram:AbＣ"


def test_platform_collision_attach_and_undo_bytes(store, ops, api):
    a, b = ops.create_entity("A"), ops.create_entity("B")
    alias = ops.add_alias(a, "ExactID", "platform_id", platform="TG", verified=True)
    assert ops.holder_for_platform_id("tg:ExactID") == a
    before = len(store.events_after())
    assert ops.attach_or_create_for_platform_id("tg:ExactID", "unused", verified=True) == a
    assert len(store.events_after()) == before
    refused(api, "unverified_platform_id", lambda: ops.add_alias(b, "x", "platform_id", platform="tg"))
    refused(api, "missing_owner", lambda: ops.add_alias(a, "my sister", "relation_phrase"))
    ops.retire_alias(alias)
    target = last(store)
    other = ops.add_alias(b, "ExactID", "platform_id", platform="tg", verified=True)
    events = store.events_after()
    error = refused(api, "dependent_events", lambda: ops.undo(target))
    assert error.blocking_event_ids == [last(store)]
    assert store.events_after() == events
    ops.undo(last(store))
    ops.undo(target)
    assert store.get_alias(other)["retired_at"] is not None
    assert store.events_after()[:len(events)] == events
    refused(api, "already_undone", lambda: ops.undo(target))
    refused(api, "is_undo", lambda: ops.undo(last(store)))
    unknown = store.append_event("forget", {})
    refused(api, "unsupported_op", lambda: ops.undo(unknown))


def test_one_event_per_op_and_void_rows(store, ops, api):
    empty = api.resolved_view(store)
    calls = [lambda: ops.create_entity("A")]
    for call in calls:
        before = len(store.events_after())
        entity = call()
        assert len(store.events_after()) == before + 1
    alias = ops.add_alias(entity, "“ ＡＬＥＸ ”")
    assert store.get_alias(alias)["alias_norm"] == "alex"
    ops.add_note("fact", entity_id=entity, cites=[1])
    ops.change_kind(entity, "thing")
    ops.retire_alias(alias)
    assert [r["op"] for r in store.events_after()] == [
        "create_entity", "add_alias", "add_note", "change_kind", "retire_alias"]
    for event in reversed(store.events_after()):
        ops.undo(event["event_id"])
        assert api.replay_projection(store) == projection(store)
    assert api.resolved_view(store) == empty
    assert store.get_entity(entity) is not None


def test_shared_alias_and_split_refusals(store, ops, api):
    a, b = ops.create_entity("A"), ops.create_entity("B")
    alias = ops.add_alias(a, "Alex")
    ops.add_alias(b, "Alex")
    assert ops.lookup_alias("Alex") == {"candidates": sorted([a, b]), "ambiguous": True}
    refused(api, "empty_selection", lambda: ops.split(a, [], [], "empty"))
    refused(api, "foreign_row", lambda: ops.split(b, [alias], [], "foreign"))
    ops.merge(a, b)
    refused(api, "not_active", lambda: ops.split(b, [], [], "merged"))
    with store.transaction():
        store.set_entity_status(a, "forgotten", log_event=False)
    refused(api, "forgotten", lambda: ops.merge(a, ops.create_entity("C")))


def test_corrupt_canonical_hop_limit(store, ops, api):
    a, b = ops.create_entity("A"), ops.create_entity("B")
    with store.transaction():
        store.set_entity_status(a, "merged", b, log_event=False)
        store.set_entity_status(b, "merged", a, log_event=False)
    with pytest.raises(api.EntityIntegrityError):
        api.resolved_view(store)


@pytest.mark.parametrize("seed", range(300))
def test_seeded_sequences(seed, store, ops, api):
    rng = random.Random(seed)
    empty = api.resolved_view(store)
    entities = [ops.create_entity(f"E{i}") for i in range(6)]
    for index in range(40):
        entity, other = rng.choices(entities, k=2)
        aliases = store.aliases_for_entity(entity)
        event = rng.choice(store.events_after())
        actions = [lambda: ops.merge(entity, other),
                   lambda: ops.change_kind(entity, rng.choice(["person", "org"])),
                   lambda: ops.add_alias(entity, rng.choice(["Alex", "Ａlex", "Sam"])),
                   lambda: ops.add_note("fact", entity_id=entity, cites=[1]),
                   lambda: ops.undo(event["event_id"]),
                   lambda: ops.add_alias(entity, "shared", "platform_id", platform="tg", verified=True)]
        notes = store.notes_for_entity(entity)
        if notes:
            note = rng.choice(notes)["note_id"]
            actions.append(lambda: entities.append(ops.split(entity, [], [note], "note split")))
        if aliases:
            alias = rng.choice(aliases)["alias_id"]
            actions.extend([lambda: ops.retire_alias(alias),
                            lambda: entities.append(ops.split(entity, [alias], [], "split"))])
        event_count = len(store.events_after())
        # R1/R2: malformed identifiers are non-mutating; exact identifier bytes
        # never re-enter the name-normalised platform-id candidate set.
        malformed = rng.choice(["abc", "tg:", ":x"])
        before = scope_rows(store)
        refused(api, "invalid_platform_id", lambda: ops.holder_for_platform_id(malformed))
        refused(api, "invalid_platform_id", lambda: ops.attach_or_create_for_platform_id(
            malformed, "invalid", verified=True))
        assert scope_rows(store) == before, (seed, index)
        exact = ops.holder_for_platform_id("tg:SHARED")
        assert exact is None, (seed, index)
        assert ops.lookup_alias("tg:SHARED")["candidates"] == [], (seed, index)
        try:
            rng.choice(actions)()
        except api.EntityOpRefused:
            assert len(store.events_after()) == event_count, (seed, index)
        else:
            assert len(store.events_after()) == event_count + 1, (seed, index)
        assert api.replay_projection(store) == projection(store), (seed, index)
        api.resolved_view(store)
        assert not store.connection.execute("SELECT 1 FROM lcm_entity_aliases WHERE alias_kind='platform_id' "
            "AND retired_at IS NULL GROUP BY scope_key,alias_norm HAVING count(*)>1").fetchall(), (seed, index)
    undone = {e["target_event_id"] for e in store.events_after() if e["op"] == "undo"}
    for event in reversed(store.events_after()):
        if event["op"] != "undo" and event["event_id"] not in undone:
            event_count = len(store.events_after())
            ops.undo(event["event_id"])
            assert len(store.events_after()) == event_count + 1, seed
            assert api.replay_projection(store) == projection(store), seed
    assert api.resolved_view(store) == empty, seed


def test_merge_and_undo_serialize(store, ops, api):
    a, b, c = [ops.create_entity(name) for name in "ABC"]
    ops.merge(a, b)
    target = last(store)
    barrier = threading.Barrier(2)

    def run(action):
        with closing(EntityStore(store.db_path, profile_key=store.scope_key)) as other:
            bound = api.EntityOps(other)
            barrier.wait(timeout=10)
            try:
                action(bound)
                return "ok"
            except api.EntityOpRefused as error:
                return error.reason

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, lambda o: o.merge(b, c)), pool.submit(run, lambda o: o.undo(target))]
        outcomes = [f.result(timeout=20) for f in futures]
    assert outcomes in [["ok", "ok"], ["ok", "dependent_events"]]
    if outcomes == ["ok", "ok"]:
        assert store.get_entity(b)["merged_into"] is None
        assert store.get_entity(c)["merged_into"] == b
    else:
        assert store.get_entity(b)["merged_into"] == a
        assert store.get_entity(c)["merged_into"] == a
    assert api.replay_projection(store) == projection(store)


def test_principal_visibility_and_merged_support(store, ops, api):
    a, b, parent = [ops.create_entity(name) for name in ("P1 Alex", "P2 Alex", "parent")]
    ops.add_alias(a, "Alex", source_store_ids=[1])
    ops.add_alias(b, "Alex", source_store_ids=[2])
    ops.add_note("hidden note", entity_id=b, cites=[1, 2])
    access = dict(access_scope="P2", principal_is_owner=False)
    assert ops.lookup_alias("Alex", **access) == {"candidates": [b], "ambiguous": False}
    view = api.resolved_view(store, **access)
    assert a not in json.dumps(view) and parent not in json.dumps(view)
    assert view["cards"][b]["notes"] == []
    ops.merge(parent, b)
    view = api.resolved_view(store, **access)
    assert view["canonical"] == {b: parent, parent: parent}
    assert view["aliases"]["alex"] == {"candidates": [parent], "ambiguous": False}
    assert view["cards"][parent]["aliases"][0]["alias_display"] == "Alex"
    assert a not in json.dumps(view)


def test_uncited_owner_empty_scope_and_strict_bool(store, ops, api):
    entity = ops.create_entity("uncited")
    ops.add_alias(entity, "Uncited")
    ops.add_note("invisible note", entity_id=entity, cites=[999])
    empty = {"canonical": {}, "aliases": {}, "cards": {}}
    assert api.resolved_view(store, "P1") == empty
    owner = api.resolved_view(store, "P1", True)
    assert owner["canonical"] == {entity: entity}
    assert owner["cards"][entity]["notes"] == []
    assert api.resolved_view(store, "", True) == empty
    assert ops.lookup_alias("Uncited", "", True) == {"candidates": [], "ambiguous": False}
    for invalid in (None, 0, 1, "true"):
        with pytest.raises(TypeError, match="must be a bool"):
            api.resolved_view(store, "P1", invalid)
        with pytest.raises(TypeError, match="must be a bool"):
            ops.lookup_alias("Uncited", "", invalid)


def test_ruling_event_sentinels_kind_and_void(store, ops, api):
    visible, hidden = ops.create_entity("visible"), ops.create_entity("hidden")
    ops.add_alias(visible, "visible", source_store_ids=[2])
    hidden_alias = ops.add_alias(hidden, "HIDDEN_ALIAS_SENTINEL", source_store_ids=[1])
    store.append_event("add_alias", {"alias_id": hidden_alias, "text": "EVENT_ALIAS_SENTINEL"})
    hidden_note = ops.add_note("HIDDEN_NOTE_SENTINEL", entity_id=hidden, cites=[1])
    store.append_event("add_note", {"note_id": hidden_note, "text": "EVENT_NOTE_SENTINEL"})
    store.append_event("change_kind", {"kind": {"entity_id": hidden, "after": "org"},
                                       "actor": "EVENT_ACTOR_SENTINEL"}, actor="EVENT_ACTOR_SENTINEL")
    access = dict(access_scope="P2", principal_is_owner=False)
    before = api.resolved_view(store, **access)
    ops.change_kind(visible, "org")
    assert api.resolved_view(store, **access)["cards"][visible]["kind"] == "org"
    ops.undo(last(store))
    assert api.resolved_view(store, **access) == before
    note = ops.add_note("visible note", entity_id=visible, cites=[2])
    ops.undo(last(store))
    alias = ops.add_alias(visible, "temporary", source_store_ids=[2])
    ops.undo(last(store))
    created = ops.create_entity("void entity")
    ops.undo(last(store))
    outputs = [api.resolved_view(store, **access), ops.lookup_alias("temporary", **access),
               api.lookup_alias(store, "HIDDEN_ALIAS_SENTINEL", **access)]
    serialized = json.dumps(outputs)
    assert "SENTINEL" not in serialized
    assert all(str(key) not in serialized for key in (hidden, hidden_note, note, created))
    assert all(a["alias_id"] != alias for c in outputs[0]["cards"].values() for a in c["aliases"])
    with pytest.raises(ValueError, match="trusted replay only"):
        store.events_after(**access)


def test_scope_mutations_and_event_reduction_are_isolated(store, ops, api):
    entity = ops.create_entity("A")
    alias = ops.add_alias(entity, "exact", "platform_id", platform="tg", verified=True)
    note = ops.add_note("A note", entity_id=entity, cites=[1])
    event = last(store)
    original = api.resolved_view(store)
    digest = scope_rows(store)
    with closing(EntityStore(store.db_path, profile_key="profile-B")) as other:
        bound = api.EntityOps(other)
        foreign = bound.create_entity("B")
        bound.add_alias(foreign, "exact", "platform_id", platform="tg", verified=True)
        bound.change_kind(foreign, "org")
        assert bound.holder_for_platform_id("tg:exact") == foreign
        assert api.resolved_view(store) == original
        before = list(other.connection.iterdump())
        for call in [lambda: bound.merge(foreign, entity), lambda: bound.merge(entity, foreign),
                     lambda: bound.split(entity, [], [], "foreign"),
                     lambda: bound.split(foreign, [alias], [], "foreign"),
                     lambda: bound.split(foreign, [], [note], "foreign"),
                     lambda: bound.undo(event), lambda: bound.retire_alias(alias),
                     lambda: bound.add_alias(entity, "foreign")]:
            foreign_error = refused(api, "not_found", call)
            assert str(foreign_error) == "not_found"
        refused(api, "not_found", lambda: bound.undo(999999))
        assert list(other.connection.iterdump()) == before
    assert scope_rows(store) == digest
    assert api.resolved_view(store) == original


def test_empty_profile_merge_missing_and_foreign_are_identical(store, ops, api):
    foreign = ops.create_entity("profile A entity")
    original = scope_rows(store)
    with closing(EntityStore(store.db_path, profile_key="empty-profile-B")) as other:
        bound = api.EntityOps(other)
        for only_void_rows in (False, True):
            if only_void_rows:
                bound.create_entity("void entity")
                bound.undo(last(other))
            assert api.resolved_view(other) == {"canonical": {}, "aliases": {}, "cards": {}}
            before, changes = scope_rows(other), other.connection.total_changes
            for into, source in [(foreign, "ent_missing"), ("ent_missing", foreign),
                                 ("ent_missing", "ent_also_missing")]:
                error = refused(api, "not_found", lambda: bound.merge(into, source))
                assert str(error) == "not_found"
                assert error.__dict__ == {"reason": "not_found"}
            assert scope_rows(other) == before
            assert other.connection.total_changes == changes
    assert scope_rows(store) == original


def test_read_reduction_cost_and_cross_instance_freshness(store, ops, api):
    entity = ops.create_entity("A")
    ops.add_alias(entity, "A", source_store_ids=[1])
    access = dict(access_scope="P1", principal_is_owner=False)
    statements = []
    store.connection.set_trace_callback(statements.append)
    try:
        with api._read_snapshot(store):
            store._read("lcm_entities")
            store._read("lcm_entity_aliases", **access)
            store._read("lcm_memory_notes", **access)
        baseline = len(statements)
        for read in (lambda: api.resolved_view(store, **access),
                     lambda: ops.lookup_alias("A", **access)):
            statements.clear()
            read()
            # W1a had four SELECTs; the required W1b key view adds one.
            assert len(statements) == baseline + 2
            assert sum(sql.lstrip().upper().startswith("SELECT") for sql in statements) == 5
            assert sum("FROM lcm_memory_events" in sql for sql in statements) == 1
            assert all("scope_key=" in sql for sql in statements if "FROM lcm_memory_events" in sql)
    finally:
        store.connection.set_trace_callback(None)
    with closing(EntityStore(store.db_path, profile_key=store.scope_key)) as other:
        api.EntityOps(other).change_kind(entity, "org")
        assert api.resolved_view(store, **access)["cards"][entity]["kind"] == "org"
    with closing(EntityStore(store.db_path, profile_key=store.scope_key, read_only=True)) as reader:
        assert api.resolved_view(reader, **access)["cards"][entity]["kind"] == "org"


def test_initial_note_projection_and_reference_touch_set(store, ops, api):
    a, b = ops.create_entity("A"), ops.create_entity("B")
    original = ops.add_note("original", entity_id=a, cites=[1])
    target = last(store)
    copied = ops.add_note("copy", entity_id=b, cites=[1], superseded_by=original, retracts_note_id=original)
    assert payload(store)["touches"]["notes"] == [original, copied]
    assert api.replay_projection(store) == projection(store)
    refused(api, "dependent_events", lambda: ops.undo(target))
    ops.undo(last(store))
    ops.undo(target)
    assert api.replay_projection(store) == projection(store)


def test_restore_first_mutant_is_killed_and_rolls_back(store, ops, api, monkeypatch):
    entity = ops.create_entity("A")
    alias = ops.add_alias(entity, "exact", "platform_id", platform="tg", verified=True)
    ops.split(entity, [alias], [], "B")
    event, before = last(store), scope_rows(store)

    def restore_first(bound, p):
        for change in reversed(p["changes"]):
            bound._project(bound._undo_payload, change["family"], change["id"], **change["before"])
        for key in p["created"]["aliases"]:
            bound._project(bound._undo_payload, "aliases", key, retired_at=True)

    with monkeypatch.context() as patch:
        patch.setitem(api.INVERSE_REGISTRY, "split", restore_first)
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            ops.undo(event)
    assert scope_rows(store) == before
    ops.undo(event)
    assert api.replay_projection(store) == projection(store)


# --- Rule 1 / W1A-2 / W1S-1 / CR-7: split with notes writes exactly one event (kills R5b).
def test_split_with_notes_is_one_event(store, ops, api):
    e = ops.create_entity("E")
    alias = ops.add_alias(e, "Alex", source_store_ids=[2])
    note = ops.add_note("fact", entity_id=e, cites=[2])
    before = len(store.events_after())
    ops.split(e, [alias], [note], "E2")
    assert len(store.events_after()) == before + 1
    assert [r["op"] for r in store.events_after()[before:]] == ["split"]


# --- Rule 6.3 / ADR D4: split supersedes the original note by the copy (kills M19).
def test_split_supersedes_original_note(store, ops, api):
    e = ops.create_entity("E")
    ops.add_alias(e, "keep", source_store_ids=[2])
    note = ops.add_note("fact", entity_id=e, cites=[2])
    e2 = ops.split(e, [], [note], "E2")
    copy = store.notes_for_entity(e2)[0]["note_id"]
    assert store.get_note(note)["superseded_by"] == copy
    view = api.resolved_view(store)
    assert view["cards"][e]["notes"] == []
    assert [n["note_id"] for n in view["cards"][e2]["notes"]] == [copy]


# --- Rule 6 refusals: a retired alias or a superseded note is foreign_row (kills M36).
def test_split_refuses_inactive_rows(store, ops, api):
    e = ops.create_entity("E")
    alias = ops.add_alias(e, "old")
    ops.add_alias(e, "keep")
    ops.retire_alias(alias)
    refused(api, "foreign_row", lambda: ops.split(e, [alias], [], "E2"))
    n1 = ops.add_note("one", entity_id=e, cites=[1])
    ops.add_note("two", entity_id=e, cites=[1], superseded_by=None)
    ops.split(e, [], [n1], "E3")  # n1 now superseded on E
    count = len(store.events_after())
    refused(api, "foreign_row", lambda: ops.split(e, [], [n1], "E4"))
    assert len(store.events_after()) == count


# --- Rule 6 + D8: split copies keep provenance, so visibility does not widen (kills M60/M62).
def test_split_copy_keeps_provenance_and_owner(store, ops, api):
    e, owner = ops.create_entity("E"), ops.create_entity("speaker")
    p1_only = ops.add_alias(e, "P1 SECRET NAME", source_store_ids=[1])
    rel = ops.add_alias(e, "my sister", "relation_phrase", owner_entity_id=owner, source_store_ids=[2])
    ops.add_alias(e, "visible", source_store_ids=[2])
    p2_owner = dict(access_scope="P2", principal_is_owner=True)
    assert "SECRET" not in json.dumps(api.resolved_view(store, **p2_owner))
    e2 = ops.split(e, [p1_only, rel], [], "E2")
    copies = {a["alias_display"]: a for a in store.aliases_for_entity(e2)}
    assert json.loads(copies["P1 SECRET NAME"]["source_store_ids"]) == [1]
    assert copies["my sister"]["owner_entity_id"] == owner
    assert "SECRET" not in json.dumps(api.resolved_view(store, **p2_owner))


# --- Rule 2: the LATEST not-undone change_kind wins (kills M1).
def test_latest_change_kind_wins(store, ops, api):
    a, b = ops.create_entity("A"), ops.create_entity("B", "org")
    ops.change_kind(a, "org")
    ops.change_kind(a, "thing")
    assert api.resolved_view(store)["cards"][a]["kind"] == "thing"
    refused(api, "kind_mismatch", lambda: ops.merge(b, a))


# --- Rule 5: allow_kind_mismatch=True merges; forgotten FROM side refused (kills M21/M22).
def test_merge_allow_kind_mismatch_and_forgotten_from(store, ops, api):
    a, b = ops.create_entity("A"), ops.create_entity("B", "org")
    refused(api, "kind_mismatch", lambda: ops.merge(a, b))
    ops.merge(a, b, allow_kind_mismatch=True)
    assert api.resolved_view(store)["canonical"][b] == a
    c, d = ops.create_entity("C"), ops.create_entity("D")
    with store.transaction():
        store.set_entity_status(d, "forgotten", log_event=False)
    refused(api, "forgotten", lambda: ops.merge(c, d))


# --- Rule 4: blocking ids newest first (kills M27).
def test_blockers_newest_first(store, ops, api):
    e = ops.create_entity("E")
    target = last(store)
    ops.add_alias(e, "x")
    first = last(store)
    ops.add_alias(e, "y")
    second = last(store)
    error = refused(api, "dependent_events", lambda: ops.undo(target))
    assert error.blocking_event_ids[:2] == [second, first]


# --- Rule 4 touch set: both canonical sides of a merge (kills M64).
def test_merge_touch_set_includes_canonical_into(store, ops, api):
    a, b, c = [ops.create_entity(n) for n in "ABC"]
    ops.merge(a, b)                      # B -> A
    ops.change_kind(a, "org")
    kind_event = last(store)
    ops.change_kind(c, "org")
    ops.merge(b, c)                      # into=B resolves to A; C -> A (kind check used A's derived kind)
    refused(api, "dependent_events", lambda: ops.undo(kind_event))


# --- Rule 9 / acceptance 5: undo's dependent scan runs inside BEGIN IMMEDIATE (kills R22).
def test_undo_scan_inside_transaction(store, ops, api):
    a, b, c = [ops.create_entity(n) for n in "ABC"]
    ops.merge(a, b)
    target = last(store)
    merged = threading.Event()
    with closing(EntityStore(store.db_path, profile_key=store.scope_key)) as s1, \
            closing(EntityStore(store.db_path, profile_key=store.scope_key)) as s2:
        undoer, merger = api.EntityOps(s1), api.EntityOps(s2)
        real = s1.transaction

        def gated():
            # The merge on the second connection commits before undo's transaction begins.
            if not merged.is_set():
                t = threading.Thread(target=lambda: (merger.merge(b, c), merged.set()))
                t.start()
                t.join(10)
            return real()

        s1.transaction = gated
        try:
            outcome = undoer.undo(target)
            result = "ok"
        except api.EntityOpRefused as error:
            result = error.reason
    assert merged.is_set()
    # Serial order is merge-then-undo, so the undo must see the merge and refuse.
    assert result == "dependent_events", (result, outcome if result == "ok" else None)


def test_hidden_merged_into_visible_never_leaks(store, ops, api):
    hidden, visible = ops.create_entity("P1"), ops.create_entity("P2")
    ops.add_alias(hidden, "Alex", source_store_ids=[1])
    ops.add_note("P1_ONLY_SENTINEL", entity_id=hidden, cites=[1])
    ops.add_alias(visible, "Alex", source_store_ids=[2])
    access = dict(access_scope="P2", principal_is_owner=False)
    before = api.resolved_view(store, **access)
    ops.merge(visible, hidden)
    outputs = [api.resolved_view(store, **access), ops.lookup_alias("Alex", **access)]
    assert outputs[0]["canonical"] == {visible: visible}
    assert outputs[0] == before
    assert hidden not in json.dumps(outputs)
    assert "SENTINEL" not in json.dumps(outputs)


def test_visible_records_hide_reference_ids(store, ops, api):
    hidden, visible = ops.create_entity("hidden"), ops.create_entity("visible")
    old = ops.add_note("hidden note", entity_id=hidden, cites=[1])
    ops.add_alias(visible, "my sister", "relation_phrase", owner_entity_id=hidden, source_store_ids=[2])
    ops.add_note("visible fact", entity_id=visible, cites=[2], about_entity_id=hidden,
                 asserted_by_entity_id=hidden, retracts_note_id=old)
    outputs = [api.resolved_view(store, "P2", True), ops.lookup_alias("my sister", "P2", True)]
    assert outputs[0]["cards"][visible]["notes"][0]["text"] == "visible fact"
    assert hidden not in json.dumps(outputs) and old not in json.dumps(outputs)


def test_other_profile_forged_events_do_not_change_projection(store, ops, api):
    entity = ops.create_entity("A")
    ops.add_alias(entity, "Alex", source_store_ids=[1])
    note = ops.add_note("fact", entity_id=entity, cites=[1])
    target = last(store)
    before = [api.resolved_view(store), api.resolved_view(store, "P1"), api.replay_projection(store)]
    with closing(EntityStore(store.db_path, profile_key="profile-B")) as other:
        other.append_event("change_kind", {"kind": {"entity_id": entity, "after": "org"}})
        other.append_event("undo", {}, target_event_id=target)
        other.append_event("create_entity", {"created": {"notes": [note]}})
    assert [api.resolved_view(store), api.resolved_view(store, "P1"), api.replay_projection(store)] == before
    ops.undo(target)
    assert api.resolved_view(store, "P1")["cards"][entity]["notes"] == []


def test_pid_five_op_shrunk_sequence_undo_all(store, ops, api):
    """Hypothesis shrink: attach, retire, attach, retire, undo(first retire)."""
    empty = api.resolved_view(store)
    first = ops.attach_or_create_for_platform_id("tg:pid", "A", verified=True)
    ops.retire_alias(store.aliases_for_entity(first)[0]["alias_id"])
    target = last(store)
    second = ops.attach_or_create_for_platform_id("tg:pid", "B", verified=True)
    attach = last(store)
    ops.retire_alias(store.aliases_for_entity(second)[0]["alias_id"])
    retirement = last(store)
    before = scope_rows(store)
    error = refused(api, "dependent_events", lambda: ops.undo(target))
    assert error.blocking_event_ids == [retirement, attach]
    assert scope_rows(store) == before
    for event in reversed(store.events_after()):
        ops.undo(event["event_id"])
        assert api.replay_projection(store) == projection(store)
    assert api.resolved_view(store) == empty


@pytest.mark.parametrize("operation", ["holder", "attach", "lookup"])
def test_pid_prefix_normalized_for_attach_holder_lookup(operation, store, ops, api):
    entity = ops.create_entity("A")
    ops.add_alias(entity, "AbC", "platform_id", platform="tg", verified=True)
    before = scope_rows(store)
    if operation == "holder":
        assert ops.holder_for_platform_id("TG:AbC") == entity
    elif operation == "attach":
        assert ops.attach_or_create_for_platform_id("TG:AbC", "unused", verified=True) == entity
    else:
        assert ops.lookup_alias("TG:AbC") == {"candidates": [entity], "ambiguous": False}
    assert ops.lookup_alias("TG:abc") == {"candidates": [], "ambiguous": False}
    assert scope_rows(store) == before


def test_pid_holder_on_read_only_store(store, ops, api):
    entity = ops.create_entity("A")
    ops.add_alias(entity, "AbC", "platform_id", platform="tg", verified=True)
    with closing(EntityStore(store.db_path, profile_key=store.scope_key, read_only=True)) as reader:
        bound = api.EntityOps(reader)
        assert bound.holder_for_platform_id("tg:AbC") == entity
        assert bound.lookup_alias("TG:AbC") == {"candidates": [entity], "ambiguous": False}
