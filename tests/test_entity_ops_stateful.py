# ruff: noqa: F811
"""D6 oracle: every intermediate answer must agree with a fresh copied-store rebuild."""
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


for name, examples in (("ci", 25), ("nightly", 500)):
    settings.register_profile(name, max_examples=examples, stateful_step_count=12,
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

    def teardown(self):
        self.stack.close()

    def attempt(self, call):
        try:
            return call()
        except self.api.EntityOpRefused:
            return None

    @rule(key=st.sampled_from(KEYS))
    def publish_key(self, key):
        self.sources.append((key, publish(self.bridge, key)))

    @rule(outcomes=GATES)
    def rebuild(self, outcomes):
        self.evaluated = {source for _, source in self.sources}
        self.gates = {source: (self.owners[g[0]], g[1], g[2])
                      for i, (_, source) in enumerate(self.sources)
                      if (g := outcomes[i % len(outcomes)]) is not None}
        self.api.rebuild_entity_keys(self.store, sender_lookup=self.gates.get)

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
        events = [e for e in self.store.events_after() if e["op"] != "undo"]
        if events:
            self.attempt(lambda: self.ops.undo(events[choice % len(events)]["event_id"]))

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

    def answer(self, store, key, source, owner):
        try:
            return self.api.resolve_key(store, key, source, owner_entity_id=owner)
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
        # SQLite backup, not a second live connection to the candidate. Reuse one
        # reference DB per example so nightly does not retain thousands of stores.
        self.store.connection.backup(self.reference.connection)
        self.api.rebuild_entity_keys(self.reference, sender_lookup=self.gates.get)
        for key, source in self.sources:
            for owner in (None, *self.owners):
                answer = self.answer(self.store, key, source, owner)
                expected = self.answer(self.reference, key, source, owner)
                assert answer is None or answer == expected, (key, source, owner, answer, expected, self.gates)


def test_entity_bridge_stateful(tmp_path):
    check_default_off(tmp_path / "default-off")
    sequence = iter(range(PROFILE.max_examples * 1000))
    run_state_machine_as_test(lambda: EntityBridgeMachine(tmp_path / f"example-{next(sequence)}"), settings=PROFILE)


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
