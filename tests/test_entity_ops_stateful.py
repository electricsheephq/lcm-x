# ruff: noqa: F811
"""D6 safety and liveness against copied and empty-derived-index rebuilds."""
from contextlib import ExitStack, closing
import importlib
import json
import os

import pytest

pytest.importorskip("hypothesis")
from hypothesis import Phase, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule, run_state_machine_as_test

from hermes_lcm.assertion_store import AssertionStore
from hermes_lcm.entity_store import EntityStore
from hermes_lcm.store import MessageStore
from tests.test_entity_bridge import bridge, publish, rebuild  # noqa: F401
from tests.test_entity_ops_identity import test_flag_off_schema_and_turn_path as check_default_off


for name, examples, steps in (("ci", 25, 20), ("nightly", 500, 30)):
    settings.register_profile(name, max_examples=examples, stateful_step_count=steps,
        derandomize=True, deadline=None, database=None,
        phases=tuple(p for p in Phase if p is not Phase.explain))
PROFILE = settings.get_profile(os.environ.get("LCM_HYPOTHESIS_PROFILE", "ci"))
KEYS = ("person:sister", "user:self", "person:alex")
CHOICE = st.integers(min_value=0, max_value=31)
GATES = st.lists(st.sampled_from((None, (0, True, True), (1, True, True),
    (0, False, True), (1, False, True), (0, True, False), (1, True, False))),
    min_size=1, max_size=8)


class EntityBridgeMachine(RuleBasedStateMachine):
    def __init__(self, root):
        super().__init__()
        self.stack = ExitStack()
        root.mkdir()
        path = root / "live.db"
        self.messages = self.stack.enter_context(closing(MessageStore(path)))
        self.messages.connection.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
        self.messages.connection.commit()
        self.assertions = self.stack.enter_context(closing(AssertionStore(path)))
        self.store = self.stack.enter_context(closing(EntityStore(path, profile_key="A")))
        self.reference = self.stack.enter_context(closing(EntityStore(root / "reference.db", profile_key="A")))
        self.independent = self.stack.enter_context(closing(EntityStore(root / "independent.db", profile_key="A")))
        self.api = importlib.import_module("hermes_lcm.entity_ops")
        self.ops = self.api.EntityOps(self.store)
        self.bridge = self.api, self.ops, self.store, self.messages, self.assertions
        self.owners = [self.ops.create_entity("owner one"), self.ops.create_entity("owner two")]
        self.targets = [self.ops.create_entity("sister one"), self.ops.create_entity("sister two")]
        self.pins = [self.ops.create_entity("pin one"), self.ops.create_entity("pin two")]
        for owner, target in zip(self.owners, self.targets):
            self.ops.add_alias(target, "my sister", "relation_phrase", owner_entity_id=owner)
        self.ops.add_alias(self.pins[0], "alex")
        self.sources, self.gates, self.seen, self.message_bytes = [], {}, {}, {}
        self.evaluated = set()
        self.resolved_at_rebuild = set()
        self.oracle_stamp = None
        self.setup_event_id = self.store.events_after()[-1]["event_id"]

    def teardown(self):
        self.stack.close()

    def attempt(self, call):
        try:
            return call()
        except self.api.EntityOpRefused:
            return None

    @rule(key=st.sampled_from(KEYS), scope=st.sampled_from(("P1", "P2", None)))
    def publish_key(self, key, scope="P1"):
        self.sources.append((key, publish(self.bridge, key, scope=scope)))

    @rule(outcomes=GATES)
    def rebuild(self, outcomes):
        self.evaluated = {source for _, source in self.sources}
        self.gates = {source: (self.owners[g[0]], g[1], g[2])
                      for i, (_, source) in enumerate(self.sources)
                      if (g := outcomes[i % len(outcomes)]) is not None}
        self.api.rebuild_entity_keys(self.store, sender_lookup=self.gates.get)
        self.check_rebuild_oracle(record_rebuild=True)

    @rule(key=st.sampled_from(KEYS), owner=st.integers(0, 2), member=st.integers(0, 1), scoped=st.booleans())
    def override(self, key, owner, member, scoped):
        owner = (None, *self.owners)[owner]
        before = len(self.store.events_after())
        self.attempt(lambda: self.ops.override_key(key, self.pins[member], owner_entity_id=owner,
            scope_store_ids=[s for k, s in self.sources if k == key] if scoped else None))
        if len(events := self.store.events_after()) == before:
            return
        override = json.loads(events[-1]["payload_json"])["override"]
        if override["global_override"] and owner is not None:
            for source in set(override["scope_store_ids"]) & self.evaluated:
                assert self.gates.get(source) == (owner, True, True), ("failed-source citation", override, self.gates)

    @rule(choice=CHOICE)
    def undo(self, choice):
        events = self.undo_candidates()
        if events:
            self.attempt(lambda: self.ops.undo(events[choice % len(events)]["event_id"]))

    @rule(member=st.integers(0, 3), kind=st.sampled_from(("person", "org", "other", "topic")))
    def change_kind(self, member, kind):
        self.attempt(lambda: self.ops.change_kind((*self.pins, *self.targets)[member], kind))

    @rule(choice=CHOICE)
    def undo_kind_change(self, choice):
        events = [e for e in self.undo_candidates() if e["op"] == "change_kind"]
        if events:
            self.attempt(lambda: self.ops.undo(events[choice % len(events)]["event_id"]))

    @rule(access_scope=st.sampled_from(("P1", "P2")), principal_is_owner=st.booleans())
    def read_scope(self, access_scope, principal_is_owner):
        self.check_rebuild_oracle(access_scope=access_scope, principal_is_owner=principal_is_owner)

    @precondition(lambda self: bool(self.sources))
    @rule(member=st.integers(0, 5), choice=CHOICE)
    def cited_alias(self, member, choice):
        """Make generated principal reads exercise visible owners and targets too."""
        source = self.sources[choice % len(self.sources)][1]
        entity = (*self.owners, *self.targets, *self.pins)[member]
        self.attempt(lambda: self.ops.add_alias(entity, "visible identity", source_store_ids=[source]))
        if member in (2, 3):
            self.attempt(lambda: self.ops.add_alias(entity, "my sister", "relation_phrase",
                owner_entity_id=self.owners[member - 2], source_store_ids=[source]))

    @rule(owner=st.integers(0, 1), member=st.integers(0, 1),
          restore=st.sampled_from(("person", "other", "topic", "undo")))
    def recompatible_override_cycle(self, owner, member, restore):
        """Generate kind restoration across a rebuild, including undo, in PR CI."""
        self.publish_key("person:sister")
        source = self.sources[-1][1]
        self.gates[source] = (self.owners[owner], True, True)
        self.change_kind(member, "person")
        self.rebuild_current_gates()
        before = len(self.store.events_after())
        self.override("person:sister", owner + 1, member, False)
        if len(self.store.events_after()) == before:
            return  # A generated ambiguous slot remains a typed refusal.
        self.change_kind(member, "org")
        self.rebuild_current_gates()
        if restore == "undo":
            self.undo_kind_change(0)
        else:
            self.change_kind(member, restore)
        self.fresh_rebuild_oracle()
        self.rebuild_current_gates()

    def undo_candidates(self):
        events = self.store.events_after()
        undone = {e["target_event_id"] for e in events if e["op"] == "undo"}
        return [e for e in reversed(events) if e["op"] != "undo"
                and e["event_id"] > self.setup_event_id and e["event_id"] not in undone][:8]

    @rule(key=st.sampled_from(KEYS[:2]), owner=st.integers(0, 1), member=st.integers(0, 1),
          failure=st.sampled_from(("none", "unverified", "reported", "dropped")))
    def g1_cycle(self, key, owner, member, failure):
        """Generate the entire G1 family without waiting for rare rule interleavings."""
        sender = self.owners[owner]
        self.publish_key(key)
        self.gates[self.sources[-1][1]] = (sender, True, True)
        self.rebuild_current_gates()
        before = len(self.store.events_after())
        self.override(key, owner + 1, member, False)
        if len(events := self.store.events_after()) == before:
            return  # Alias ambiguity or an unavailable canonical entity is a typed refusal.
        event_id = events[-1]["event_id"]
        self.publish_key(key)
        source = self.sources[-1][1]
        self.gates[source] = (sender, True, True)
        self.rebuild_current_gates()
        self.attempt(lambda: self.ops.undo(event_id))
        self.fresh_rebuild_oracle()
        if failure != "none":
            self.gates.pop(source)
            if failure != "dropped":
                self.gates[source] = (sender, failure != "unverified", failure != "reported")
            self.rebuild_current_gates()
        self.override(key, owner + 1, 1 - member, False)
        self.fresh_rebuild_oracle()

    def rebuild_current_gates(self):
        self.evaluated = {source for _, source in self.sources}
        self.api.rebuild_entity_keys(self.store, sender_lookup=self.gates.get)
        self.check_rebuild_oracle(record_rebuild=True)

    @rule(owner=st.integers(0, 1), member=st.integers(0, 1), remove=st.booleans())
    def relation_alias(self, owner, member, remove):
        if remove:
            aliases = self.store.aliases_for_entity(self.targets[member])
            for alias in aliases:
                if alias["alias_kind"] == "relation_phrase" and alias["owner_entity_id"] == self.owners[owner]:
                    self.attempt(lambda: self.ops.retire_alias(alias["alias_id"]))
                    break
        else:
            self.attempt(lambda: self.ops.add_alias(self.targets[member], "my sister", "relation_phrase",
                                                   owner_entity_id=self.owners[owner]))

    @rule(reverse=st.booleans())
    def merge(self, reverse):
        a, b = self.targets[::(-1 if reverse else 1)]
        self.attempt(lambda: self.ops.merge(a, b))

    @rule(member=st.integers(0, 1))
    def split(self, member):
        aliases = self.store.aliases_for_entity(self.targets[member])
        if aliases:
            self.attempt(lambda: self.ops.split(self.targets[member], [aliases[0]["alias_id"]], [], "split sister"))

    def answer(self, store, key, source, owner, access_scope=None, principal_is_owner=False):
        try:
            return self.api.resolve_key(store, key, source, owner_entity_id=owner,
                access_scope=access_scope, principal_is_owner=principal_is_owner)
        except self.api.EntityOpRefused as error:
            assert error.reason == "not_found"
            return None

    @invariant()
    def retained_rows_and_default_off_bytes(self):
        for table, key in (("lcm_entities", "entity_id"), ("lcm_entity_aliases", "alias_id"),
                           ("lcm_memory_notes", "note_id"), ("lcm_entity_keys", "key_id"),
                           ("lcm_memory_events", "event_id")):
            rows = {r[0] for r in self.store.connection.execute(f"SELECT {key} FROM {table}")}
            assert self.seen.get(table, set()) <= rows, ("row deleted", table)
            self.seen[table] = rows
        current = {r[0]: tuple(r) for r in self.store.connection.execute("SELECT * FROM messages")}
        assert all(current.get(k) == row for k, row in self.message_bytes.items()), "message bytes changed"
        self.message_bytes = current

    @precondition(lambda self: bool(self.sources))
    @invariant()
    def fresh_rebuild_oracle(self):
        self.check_rebuild_oracle()

    def check_rebuild_oracle(self, record_rebuild=False, access_scope=None, principal_is_owner=False):
        stamp = self.oracle_inputs()
        if access_scope is None and not record_rebuild and stamp == self.oracle_stamp:
            return  # Every fixture writer and sender/source input is unchanged.
        events = self.store.events_after()
        undone = {e["target_event_id"] for e in events if e["op"] == "undo"}
        overridden = {(p["subject_key"], p["owner_entity_id"]) for e in events
                      if e["op"] == "override_key" and e["event_id"] not in undone
                      for p in [json.loads(e["payload_json"])["override"]]}
        resolved = set()
        observed = {(key, source, owner): self.answer(self.store, key, source, owner, access_scope, principal_is_owner)
                    for key, source in self.sources for owner in (None, *self.owners)}
        for label, reference in (("copied", self.reference), ("empty-index", self.independent)):
            # Only disposable reference copies lose derived rows. Curated entities,
            # aliases, assertions and override/undo events remain the same inputs.
            self.store.connection.backup(reference.connection)
            if reference is self.independent:
                reference.connection.execute("DELETE FROM lcm_entity_keys")
                reference.connection.commit()
            self.api.rebuild_entity_keys(reference, sender_lookup=self.gates.get)
            view = self.api.resolved_view(reference) if record_rebuild else None
            for key, source in self.sources:
                for owner in (None, *self.owners):
                    answer = observed[key, source, owner]
                    expected = self.answer(reference, key, source, owner, access_scope, principal_is_owner)
                    detail = (label, key, source, owner, answer, expected, self.gates)
                    assert answer is None or answer == expected, ("safety", detail)
                    if record_rebuild and expected is not None and (
                            key.partition(":")[0] in {"user", "assistant"} or
                            len(self.api._key_candidates(view, key, owner)) == 1):
                        resolved.add((key, owner))
                    # R5-Q: ambiguity at the last rebuild admits no sender fact.
                    # Safety remains strict even for those unadmitted slots.
                    if access_scope is None and (key, owner) not in overridden and expected is not None and (
                            record_rebuild or (key, owner) in self.resolved_at_rebuild):
                        assert answer == expected, ("liveness", detail)
        if record_rebuild:
            self.resolved_at_rebuild = resolved
        if access_scope is None:
            self.oracle_stamp = self.oracle_inputs()

    def oracle_inputs(self):
        return (tuple(s.connection.total_changes for s in (self.store, self.messages, self.assertions)),
                tuple(self.sources), tuple(sorted(self.gates.items())), frozenset(self.resolved_at_rebuild))



def test_entity_bridge_stateful(tmp_path):
    check_default_off(tmp_path / "default-off")
    sequence = iter(range(PROFILE.max_examples * 1000))
    run_state_machine_as_test(lambda: EntityBridgeMachine(tmp_path / f"example-{next(sequence)}"), settings=PROFILE)


@pytest.mark.parametrize("key", KEYS[:2])
@pytest.mark.parametrize("failure", ["none", "unverified", "reported", "dropped"])
def test_forced_g1_cycle(tmp_path, key, failure):
    """The generated macro's deterministic seed includes the no-rebuild sibling."""
    machine = EntityBridgeMachine(tmp_path / "forced")
    try:
        machine.g1_cycle(key, 0, 0, failure)
        events = machine.store.events_after()
        assert [e["op"] for e in events if e["event_id"] > machine.setup_event_id] == [
            "override_key", "undo", "override_key"]
        assert all(e["event_id"] > machine.setup_event_id for e in machine.undo_candidates())
        machine.retained_rows_and_default_off_bytes()
    finally:
        machine.teardown()


def test_scoped_override_after_empty_rebuild_cites_only_named_source_and_owner(bridge):
    """Shrunk machine case: publish; rebuild(dropped); scoped override(owner)."""
    api, ops, store, _, _ = bridge
    owner, sister, pin = [ops.create_entity(name) for name in ("owner", "sister", "pin")]
    ops.add_alias(sister, "my sister", "relation_phrase", owner_entity_id=owner)
    source = publish(bridge, "person:sister")
    before = list(store.connection.iterdump())
    assert rebuild(bridge, sender_lookup={}.get) == 0
    assert list(store.connection.iterdump()) == before  # No durable gate snapshot exists.
    ops.override_key("person:sister", pin, owner_entity_id=owner, scope_store_ids=[source])
    override = json.loads(store.events_after()[-1]["payload_json"])["override"]
    assert override["scope_store_ids"] == [source]
    sibling = publish(bridge, "person:sister")
    other_owner = ops.create_entity("other owner")
    assert api.resolve_key(store, "person:sister", source, owner_entity_id=owner) == pin
    assert api.resolve_key(store, "person:sister", sibling, owner_entity_id=owner) is None
    assert api.resolve_key(store, "person:sister", source, owner_entity_id=other_owner) is None
    assert api.resolve_key(store, "person:sister", source) is None
    rebuild(bridge, sender_lookup={}.get)
    assert api.resolve_key(store, "person:sister", source, owner_entity_id=owner) == pin


@pytest.mark.parametrize("case", ["last-support-retired", "ambiguity-removed"])
def test_r5q_relation_alias_safety_and_rebuild_liveness(tmp_path, case):
    """F1 is immediate; previously ambiguous F2 waits for admitted sender evidence."""
    machine = EntityBridgeMachine(tmp_path / "r5p")
    try:
        machine.g1_cycle("person:sister", 0, 0, "none")
        if case == "last-support-retired":
            machine.g1_cycle("person:sister", 0, 0, "none")
            machine.relation_alias(1, 1, True)
            machine.relation_alias(0, 0, True)
            for _, source in machine.sources:
                assert machine.answer(machine.store, "person:sister", source, machine.owners[0]) is None
        else:
            machine.relation_alias(1, 0, False)
            machine.g1_cycle("person:sister", 1, 0, "none")
            machine.relation_alias(1, 0, True)
            source = machine.sources[-1][1]
            assert machine.answer(machine.store, "person:sister", source, machine.owners[1]) is None
            machine.fresh_rebuild_oracle()
            machine.rebuild_current_gates()
            assert machine.answer(machine.store, "person:sister", source, machine.owners[1]) == machine.targets[1]
            machine.gates.pop(source)
            machine.rebuild_current_gates()
            assert machine.answer(machine.store, "person:sister", source, machine.owners[1]) is None
        machine.fresh_rebuild_oracle()
        machine.retained_rows_and_default_off_bytes()
    finally:
        machine.teardown()


def test_r5q_scoped_override_does_not_admit_an_ambiguous_slot(tmp_path):
    machine = EntityBridgeMachine(tmp_path / "ambiguous-scoped")
    try:
        key, owner = "person:sister", machine.owners[0]
        machine.relation_alias(0, 1, False)
        machine.publish_key(key)
        source = machine.sources[-1][1]
        machine.gates[source] = (owner, True, True)
        machine.rebuild_current_gates()
        machine.override(key, 1, 0, True)
        event = machine.store.events_after()[-1]["event_id"]
        machine.rebuild_current_gates()
        assert (key, owner) not in machine.resolved_at_rebuild
        machine.ops.undo(event)
        machine.relation_alias(0, 1, True)
        assert machine.answer(machine.store, key, source, owner) is None
        machine.fresh_rebuild_oracle()
        machine.rebuild_current_gates()
        assert machine.answer(machine.store, key, source, owner) == machine.targets[0]
        machine.retained_rows_and_default_off_bytes()
    finally:
        machine.teardown()


@pytest.mark.parametrize("restore", ["person", "other", "topic", "undo"])
def test_forced_recompatible_override_cycle(tmp_path, restore):
    """The generated R2 cycle must reach the same R5-Q safety oracle as F1."""
    machine = EntityBridgeMachine(tmp_path / "recompatible")
    try:
        machine.recompatible_override_cycle(0, 0, restore)
        machine.retained_rows_and_default_off_bytes()
    finally:
        machine.teardown()


@pytest.mark.parametrize("visible_pin", [False, True])
@pytest.mark.parametrize("principal_is_owner", [False, True])
def test_forced_scoped_recompatible_reads(tmp_path, visible_pin, principal_is_owner):
    """Exercise the scoped oracle with visible/hidden targets and source rows."""
    machine = EntityBridgeMachine(tmp_path / "scoped-recompatible")
    try:
        machine.recompatible_override_cycle(0, 0, "undo")
        machine.cited_alias(0, 0)  # The owner is visible through the P1 source.
        machine.cited_alias(2, 0)  # So is the relation alias used by the fallback.
        if visible_pin:
            machine.cited_alias(4, 0)
        machine.read_scope("P1", principal_is_owner)
        machine.read_scope("P2", principal_is_owner)
        key, source = machine.sources[0]
        expected = machine.pins[0] if visible_pin or principal_is_owner else machine.targets[0]
        assert machine.answer(machine.store, key, source, machine.owners[0], "P1", principal_is_owner) == expected
        assert machine.answer(machine.store, key, source, machine.owners[0], "P2", principal_is_owner) is None
    finally:
        machine.teardown()
