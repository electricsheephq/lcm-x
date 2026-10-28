"""Reviewer probe (redteam-design lens): extended seeded harness for PR #1920.

Adds ops the PR harness omits: note splits, add_note with superseded_by/retracts,
relation_phrase aliases, attach_or_create, mixed cites (P1/P2/uncited), and checks
principal views for hidden-id leaks plus immediate-undo view restoration.
"""
import importlib
import json
import random
from contextlib import closing

import pytest

from hermes_lcm.entity_store import EntityStore
from hermes_lcm.store import MessageStore

api = importlib.import_module("hermes_lcm.entity_ops")

STATS = {}


PRINCIPALS = [("P1", False), ("P2", False), ("P1", True), ("P2", True), ("", True)]


def make_store(tmp_path, key="profile-A"):
    path = tmp_path / "lcm.db"
    if not path.exists():
        with closing(MessageStore(path)) as messages:
            messages.connection.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
            for k, scope in [(1, "P1"), (2, "P2"), (3, "P1"), (4, "P2")]:
                messages.connection.execute(
                    "INSERT INTO messages(store_id,session_id,role,content,timestamp,access_scope) "
                    "VALUES(?,'S','user','synthetic',1,?)", (k, scope))
            messages.connection.commit()
    return EntityStore(path, profile_key=key)


def projection(store):
    result = {}
    for fam, table, key, fields in [("entities", "lcm_entities", "entity_id", "status,merged_into"),
                                    ("aliases", "lcm_entity_aliases", "alias_id", "retired_at"),
                                    ("notes", "lcm_memory_notes", "note_id", "superseded_by")]:
        result[fam] = {row[0]: tuple(row)[1:] for row in store.connection.execute(
            f"SELECT {key},{fields} FROM {table} WHERE scope_key=?", (store.scope_key,))}
    return result


def views(store):
    out = {"trusted": api.resolved_view(store)}
    for scope, owner in PRINCIPALS:
        out[(scope, owner)] = api.resolved_view(store, scope, owner)
    return out


def supported_ids(store, scope, owner):
    """Independent oracle: entity ids a principal may see (own or merged-in visible rows)."""
    events = store.events_after()
    undone = {e["target_event_id"] for e in events if e["op"] == "undo"}
    void = set()
    for e in events:
        if e["event_id"] in undone:
            p = json.loads(e["payload_json"])
            for ids in p.get("created", {}).values():
                void.update(ids)
    ents = {r["entity_id"]: dict(r) for r in store.connection.execute(
        "SELECT * FROM lcm_entities WHERE scope_key=?", (store.scope_key,)) if r["entity_id"] not in void}
    access = dict(access_scope=scope, principal_is_owner=owner)
    rows = [r for r in store._read("lcm_entity_aliases", **access)
            if r["alias_id"] not in void and r["retired_at"] is None]
    rows += [r for r in store._read("lcm_memory_notes", **access)
             if r["note_id"] not in void and r["superseded_by"] is None]
    vis_alias = {r["alias_id"] for r in rows if "alias_id" in r}
    vis_note = {r["note_id"] for r in rows if "note_id" in r}
    sup = set()
    for r in rows:
        k = r["entity_id"]
        while k is not None and k in ents:
            sup.add(k)
            k = ents[k]["merged_into"]
    return sup, vis_alias, vis_note, set(ents), void


def check_principal(store, view, scope, owner, seed, index):
    sup, vis_alias, vis_note, all_ents, void = supported_ids(store, scope, owner)
    text = json.dumps(view)
    for eid in all_ents | {e for e in void if str(e).startswith("ent_")}:
        if eid not in sup:
            assert eid not in text, ("hidden entity leaked", seed, index, scope, owner, eid)
    for card in view["cards"].values():
        for a in card["aliases"]:
            assert a["alias_id"] in vis_alias, ("hidden alias leaked", seed, index, scope, owner)
        for n in card["notes"]:
            assert n["note_id"] in vis_note, ("hidden note leaked", seed, index, scope, owner)


@pytest.mark.parametrize("seed", range(200))
def test_extended_seeded(seed, tmp_path):
    rng = random.Random(seed)
    with closing(make_store(tmp_path)) as store:
        ops = api.EntityOps(store)
        empty = views(store)
        entities = [ops.create_entity(f"E{i}") for i in range(5)]
        for index in range(35):
            entity, other, third = rng.choices(entities, k=3)
            aliases = store.aliases_for_entity(entity)
            notes = [n for n in store.notes_for_entity(entity) if n["superseded_by"] is None]
            cite = rng.choice([[1], [2], [1, 2], [3], [4]])
            src = rng.choice([[], [1], [2], [3, 4]])
            events = store.events_after()
            event = rng.choice(events)
            actions = [
                ("merge", lambda: ops.merge(entity, other)),
                ("kind", lambda: ops.change_kind(entity, rng.choice(["person", "org", "thing"]))),
                ("alias", lambda: ops.add_alias(entity, rng.choice(["Alex", "Ａlex", "Sam"]), source_store_ids=src)),
                ("rel", lambda: ops.add_alias(entity, "my sister", "relation_phrase", owner_entity_id=third,
                                              source_store_ids=src)),
                ("note", lambda: ops.add_note(f"fact SENTINEL{cite}", entity_id=entity, cites=cite,
                                              about_entity_id=third)),
                ("undo", lambda: ops.undo(event["event_id"])),
                ("pid", lambda: ops.add_alias(entity, rng.choice(["shared", "Shared"]), "platform_id",
                                              platform="tg", verified=True, source_store_ids=src)),
                ("attach", lambda: ops.attach_or_create_for_platform_id(
                    rng.choice(["tg:shared", "tg:other"]), "attached", verified=True)),
            ]
            if notes:
                n = rng.choice(notes)["note_id"]
                actions.append(("note_sup", lambda: ops.add_note("successor", entity_id=other, cites=cite,
                                                                 superseded_by=n, retracts_note_id=n)))
            if aliases or notes:
                sel_a = [a["alias_id"] for a in aliases if rng.random() < 0.6]
                sel_n = [x["note_id"] for x in notes if rng.random() < 0.6]
                actions.append(("split", lambda: entities.append(
                    ops.split(entity, sel_a, sel_n, "split", kind=rng.choice([None, "org"])))))
            if aliases:
                alias = rng.choice(aliases)["alias_id"]
                actions.append(("retire", lambda: ops.retire_alias(alias)))
            before_views = views(store)
            count = len(store.events_after())
            name, action = rng.choice(actions)
            try:
                result = action()
            except api.EntityOpRefused as _err:
                STATS[(name, _err.reason)] = STATS.get((name, _err.reason), 0) + 1
                assert len(store.events_after()) == count, (seed, index, name)
                assert views(store) == before_views, (seed, index, name, "refusal changed view")
                ok = False
            else:
                grew = len(store.events_after()) - count
                if name == "attach" and grew == 0:
                    ok = False
                else:
                    assert grew == 1, (seed, index, name, grew)
                    ok = True
                    STATS[(name, "ok")] = STATS.get((name, "ok"), 0) + 1
                if name == "attach" and isinstance(result, str) and result.startswith("ent_") and result not in entities:
                    entities.append(result)
            assert api.replay_projection(store) == projection(store), (seed, index, name)
            assert not store.connection.execute(
                "SELECT 1 FROM lcm_entity_aliases WHERE alias_kind='platform_id' AND retired_at IS NULL "
                "GROUP BY scope_key,alias_norm HAVING count(*)>1").fetchall()
            current = views(store)
            for scope, owner in PRINCIPALS:
                check_principal(store, current[(scope, owner)], scope, owner, seed, index)
            # Immediate undo of a just-applied non-undo op must restore every view.
            if ok and name != "undo" and rng.random() < 0.35:
                last = store.events_after()[-1]["event_id"]
                try:
                    ops.undo(last)
                except api.EntityOpRefused as error:
                    raise AssertionError(("immediate undo refused", seed, index, name, error.reason,
                                          error.__dict__))
                after = views(store)
                assert after == before_views, (seed, index, name, "immediate undo view mismatch")
                assert api.replay_projection(store) == projection(store), (seed, index, name)
        undone = {e["target_event_id"] for e in store.events_after() if e["op"] == "undo"}
        for event in reversed(store.events_after()):
            if event["op"] != "undo" and event["event_id"] not in undone:
                try:
                    ops.undo(event["event_id"])
                except api.EntityOpRefused as error:
                    raise AssertionError(("undo-all refused", seed, event["op"], error.reason, error.__dict__))
                assert api.replay_projection(store) == projection(store), seed
        assert views(store) == empty, seed
