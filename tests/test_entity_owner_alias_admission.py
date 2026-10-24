# ruff: noqa: F811
"""Guard verifier probes (r3): _owner_alias_rows admission is slot-level; check observable reads."""
import json
import pytest
from tests.test_entity_bridge import bridge, publish, active, rebuild  # noqa: F401

OK, UNVER, NOTFP = (True, True), (False, True), (True, False)


def setup(bridge, namespace):
    api, ops, store, _, _ = bridge
    o1 = ops.create_entity("owner one")
    t1 = o1 if namespace == "user" else ops.create_entity("sister one")
    key = "user:self" if namespace == "user" else "person:sister"
    ops.add_alias(t1, "my sister", "relation_phrase", owner_entity_id=o1)
    pin = ops.create_entity("pinned")
    return api, ops, store, o1, t1, key, pin


def senders(o, **m):
    return lambda s: (o, *m[s]) if s in m else None


def rb(bridge, o1, mapping):
    rebuild(bridge, sender_lookup=lambda s: (o1, *mapping[s]) if s in mapping else None)


def last_override_cited(store):
    return json.loads(store.events_after()[-1]["payload_json"])["override"]["scope_store_ids"]


def reads(api, store, key, o1, *sources):
    return tuple(api.resolve_key(store, key, s, owner_entity_id=o1) for s in sources)


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_override_first_then_uncited_sibling(bridge, namespace):
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    a = publish(bridge, key)
    rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    assert last_override_cited(store) == [a]
    b = publish(bridge, key)
    rb(bridge, o1, {a: OK, b: OK})
    assert reads(api, store, key, o1, a, b) == (pin, t1)
    for gate in (UNVER, NOTFP, None):
        m = {a: OK} if gate is None else {a: OK, b: gate}
        rb(bridge, o1, m)
        assert reads(api, store, key, o1, a, b) == (pin, None), gate
        rb(bridge, o1, {a: OK, b: OK})
        assert reads(api, store, key, o1, a, b) == (pin, t1), gate


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_evidence_first_then_override_swapped_order(bridge, namespace):
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    a, b = publish(bridge, key), publish(bridge, key)
    rb(bridge, o1, {a: OK, b: UNVER})          # b fails gate before the override
    ops.override_key(key, pin, owner_entity_id=o1)
    assert last_override_cited(store) == [a]   # b never cited
    assert reads(api, store, key, o1, a, b) == (pin, None)
    rb(bridge, o1, {a: OK, b: OK})
    assert reads(api, store, key, o1, a, b) == (pin, t1)
    rb(bridge, o1, {a: OK, b: UNVER})
    assert reads(api, store, key, o1, a, b) == (pin, None)
    rb(bridge, o1, {a: OK})                    # b's sender dropped
    assert reads(api, store, key, o1, a, b) == (pin, None)


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("final", ["b_ok", "b_unver", "b_dropped"])
def test_order_independence_of_final_reads(bridge, namespace, final):
    fin = {"b_ok": OK, "b_unver": UNVER, "b_dropped": None}[final]
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    # Slot A: override first, sibling later.  Slot B (second key/owner in same store): evidence first.
    a = publish(bridge, key)
    rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    b = publish(bridge, key)
    rb(bridge, o1, {a: OK, b: OK})
    rb(bridge, o1, {a: OK, b: UNVER})
    rb(bridge, o1, {a: OK} if fin is None else {a: OK, b: fin})
    first = reads(api, store, key, o1, a, b)
    expected = (pin, t1 if fin == OK else None)
    assert first == expected
    # Same final sender map reached via a different rebuild history must read the same.
    rb(bridge, o1, {a: OK, b: OK})
    rb(bridge, o1, {a: OK} if fin is None else {a: OK, b: fin})
    rb(bridge, o1, {a: OK} if fin is None else {a: OK, b: fin})
    assert reads(api, store, key, o1, a, b) == expected


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_other_owner_override_does_not_admit_this_slot(bridge, namespace):
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    o2 = ops.create_entity("owner two")
    t2 = o2 if namespace == "user" else ops.create_entity("sister two")
    ops.add_alias(t2, "my sister", "relation_phrase", owner_entity_id=o2)
    a, c = publish(bridge, key), publish(bridge, key)
    look = {a: (o1, True, True), c: (o2, True, True)}
    rebuild(bridge, sender_lookup=look.get)
    ops.override_key(key, pin, owner_entity_id=o2)      # o2 slot gets an active override citing c
    assert last_override_cited(store) == [c]
    assert api.resolve_key(store, key, a, owner_entity_id=o1) == t1
    rebuild(bridge, sender_lookup={a: (o1, False, True), c: (o2, True, True)}.get)
    assert api.resolve_key(store, key, a, owner_entity_id=o1) is None
    assert api.resolve_key(store, key, c, owner_entity_id=o2) == pin


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("b_gate", ["ok", "unver"])
def test_undo_after_uncited_sibling(bridge, namespace, b_gate):
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    a = publish(bridge, key)
    rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    b = publish(bridge, key)
    rb(bridge, o1, {a: OK, b: OK})
    if b_gate == "unver":
        rb(bridge, o1, {a: OK, b: UNVER})
    ops.undo(ev)
    ra, rbv = reads(api, store, key, o1, a, b)
    assert ra in (t1, None) and ra != pin
    if b_gate == "unver":
        assert rbv is None, (ra, rbv, active(store))
    else:
        assert rbv in (t1, None)
    rb(bridge, o1, {a: OK, b: OK if b_gate == "ok" else UNVER})
    assert reads(api, store, key, o1, a, b) == (t1, t1 if b_gate == "ok" else None)


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_cited_source_fails_uncited_survives_then_undo(bridge, namespace):
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    a = publish(bridge, key)
    rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    b = publish(bridge, key)
    rb(bridge, o1, {a: UNVER, b: OK})          # the cited source fails; override unbound
    assert reads(api, store, key, o1, a, b) == (None, t1)
    with pytest.raises(api.EntityOpRefused, match="key_held"):
        ops.undo(ev)                           # pre-existing fail-closed refusal (same at 2ec6ac65)
    assert reads(api, store, key, o1, a, b) == (None, t1), active(store)
    ops.override_key(key, pin, owner_entity_id=o1)
    assert a not in last_override_cited(store)


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_guard_is_slot_level_characterization(bridge, namespace):
    """Documents the unchanged guard: with any active cited override in the slot, retired
    alias rows citing a gate-failed uncited source are still returned (only [:1] is read)."""
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    a = publish(bridge, key)
    rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    b = publish(bridge, key)
    rb(bridge, o1, {a: OK, b: OK})
    rb(bridge, o1, {a: OK, b: UNVER})
    rows = api._owner_alias_rows(store, key, o1)
    srcs = [json.loads(r["source_store_ids"]) for r in rows]
    # newest admitted row is refreshed to current verified sources only
    assert b not in srcs[0] and a in srcs[0], srcs
    # but older admitted (retired) rows still cite b: admission is not per store_id
    assert any(b in s for s in srcs[1:]), srcs
    assert all(r["retired_at"] is not None for r in rows)


@pytest.mark.parametrize("namespace", ["person", "user"])
@pytest.mark.parametrize("between", ["none", "b_fails", "b_dropped"])
def test_undo_then_reoverride_admits_only_current_newest(bridge, namespace, between):
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    a = publish(bridge, key)
    rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    b = publish(bridge, key)
    rb(bridge, o1, {a: OK, b: OK})            # retired shadow now cites [a, b]
    ops.undo(ev)
    if between == "b_fails":
        rb(bridge, o1, {a: OK, b: UNVER})
    elif between == "b_dropped":
        rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    cited = last_override_cited(store)
    assert b not in cited or between == "none", cited
    ra, rbv = reads(api, store, key, o1, a, b)
    assert ra == pin
    if between != "none":
        assert rbv is None, (rbv, [(r["entity_id"], r["method"], r["source_store_ids"]) for r in active(store)])
    else:
        assert rbv in (t1, pin)


@pytest.mark.parametrize("namespace", ["person", "user"])
def test_stacked_override_after_undo_never_cites_gate_failed_source(bridge, namespace):
    api, ops, store, o1, t1, key, pin = setup(bridge, namespace)
    pin2 = ops.create_entity("pinned two")
    a = publish(bridge, key)
    rb(bridge, o1, {a: OK})
    ops.override_key(key, pin, owner_entity_id=o1)
    ev = store.events_after()[-1]["event_id"]
    b = publish(bridge, key)
    rb(bridge, o1, {a: OK, b: OK})
    ops.undo(ev)
    rb(bridge, o1, {a: OK, b: UNVER})
    ops.override_key(key, pin, owner_entity_id=o1)
    assert b not in last_override_cited(store)
    ops.override_key(key, pin2, owner_entity_id=o1)      # stacked, no rebuild in between
    cited = last_override_cited(store)
    assert b not in cited, f"stacked global override cites gate-failed source: {cited}"
