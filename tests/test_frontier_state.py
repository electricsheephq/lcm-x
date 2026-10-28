"""Characterize the legacy frontier attributes and engine copy boundaries."""

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


DEFAULTS: dict[str, Any] = {
    "_last_compacted_store_id": 0,
    "_ingest_cursor": 0,
    "_ingest_cursor_needs_reconcile": False,
    "_compress_commit_proof": None,
    "_last_emission_descriptors": None,
}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Any]:
    instance = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        yield instance
    finally:
        instance.shutdown()


def test_initial_values_and_assignment_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, Any]] = []

    def record(instance: Any, name: str, value: Any) -> None:
        if name in DEFAULTS:
            writes.append((name, value))
        object.__setattr__(instance, name, value)

    monkeypatch.setattr(LCMEngine, "__setattr__", record)
    instance = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        assert writes == list(DEFAULTS.items())
        for name, value in DEFAULTS.items():
            assert getattr(instance, name) is value
    finally:
        instance.shutdown()


@pytest.mark.parametrize("name,values", [
    ("_ingest_cursor", [0, 1001, -2]),
    ("_ingest_cursor_needs_reconcile", [False, True, False]),
    ("_last_compacted_store_id", [0, 2002, -3]),
    ("_compress_commit_proof", [None, {}, {"consulted": False}, None]),
    ("_last_emission_descriptors", [None, {}, {"rows": []}, None]),
])
def test_legacy_assignment_keeps_value_type_and_identity(
    engine: Any, name: str, values: list[Any],
) -> None:
    for value in values:
        setattr(engine, name, value)
        assert getattr(engine, name) is value
        assert type(getattr(engine, name)) is type(value)


def test_dict_mutations_and_dynamic_clear_keep_identity(engine: Any) -> None:
    proof: dict[str, Any] = {"consulted": False}
    descriptors: dict[str, Any] = {"rows": []}
    engine._compress_commit_proof = proof
    engine._last_emission_descriptors = descriptors
    engine._compress_commit_proof["consulted"] = True
    engine._compress_commit_proof["end_consumed"] = True
    engine._last_emission_descriptors["rows"].append("row")
    assert proof == {"consulted": True, "end_consumed": True}
    assert descriptors == {"rows": ["row"]}
    for name in ("_compress_commit_proof", "_last_emission_descriptors"):
        setattr(engine, name, None)
        assert getattr(engine, name) is None


@pytest.mark.parametrize("mode", ["clone", "deepcopy"])
def test_fresh_engines_reset_frontier_without_sharing(engine: Any, mode: str) -> None:
    engine._ingest_cursor = 1001
    engine._ingest_cursor_needs_reconcile = True
    engine._last_compacted_store_id = 2002
    engine._compress_commit_proof = {"parent": True}
    engine._last_emission_descriptors = {"parent": True}
    clone = engine.clone_for_agent() if mode == "clone" else copy.deepcopy(engine)
    try:
        for name, value in DEFAULTS.items():
            assert getattr(clone, name) is value
        clone._compress_commit_proof = {"child": True}
        clone._last_emission_descriptors = {"child": True}
        assert engine._compress_commit_proof == {"parent": True}
        assert engine._last_emission_descriptors == {"parent": True}
        assert engine._ingest_cursor == 1001
    finally:
        clone.shutdown()


def test_shallow_copy_matches_default_dict_and_aliases(engine: Any) -> None:
    engine._ingest_cursor = 1001
    engine._last_compacted_store_id = 2002
    engine._compress_commit_proof = {"consulted": False}
    engine._last_emission_descriptors = {"rows": []}
    # The default shallow-copy state transfer, independent of __copy__.
    default = object.__new__(type(engine))
    default.__dict__.update(engine.__dict__)
    clone = copy.copy(engine)
    assert clone.__dict__.keys() == default.__dict__.keys()
    for key, value in default.__dict__.items():
        if key != "_frontier":
            assert clone.__dict__[key] is value
    for name in DEFAULTS:
        assert getattr(clone, name) is getattr(engine, name)
    clone._ingest_cursor = 3003
    clone._last_compacted_store_id = 4004
    clone._ingest_cursor_needs_reconcile = True
    assert engine._ingest_cursor == 1001
    assert engine._last_compacted_store_id == 2002
    assert engine._ingest_cursor_needs_reconcile is False
    clone._compress_commit_proof["consulted"] = True
    clone._last_emission_descriptors["rows"].append("row")
    assert engine._compress_commit_proof == {"consulted": True}
    assert engine._last_emission_descriptors == {"rows": ["row"]}
    clone._compress_commit_proof = None
    clone._last_emission_descriptors = None
    assert engine._compress_commit_proof == {"consulted": True}
    assert engine._last_emission_descriptors == {"rows": ["row"]}


@pytest.mark.parametrize("name", DEFAULTS)
@pytest.mark.parametrize("factory", [LCMEngine.__new__, object.__new__])
def test_uninitialized_engine_missing_then_set(name: str, factory: Any) -> None:
    instance = factory(LCMEngine)
    sentinel = object()
    assert not hasattr(instance, name)
    assert getattr(instance, name, sentinel) is sentinel
    with pytest.raises(AttributeError):
        getattr(instance, name)
    setattr(instance, name, DEFAULTS[name])
    assert hasattr(instance, name)
    assert getattr(instance, name) is DEFAULTS[name]


def test_shallow_copy_of_uninitialized_engine_keeps_missing_names() -> None:
    instance = object.__new__(LCMEngine)
    clone = copy.copy(instance)
    assert clone.__dict__ == instance.__dict__ == {}
    for name in DEFAULTS:
        assert not hasattr(clone, name)


def test_engine_has_no_custom_pickle_hooks_or_slots() -> None:
    for cls in LCMEngine.__mro__:
        if cls is not object:
            assert not {"__getstate__", "__setstate__", "__reduce__", "__reduce_ex__"}.intersection(vars(cls))
            assert not vars(cls).get("__slots__")


@pytest.mark.parametrize("mode", ["clone", "deepcopy", "shallow"])
def test_copies_own_distinct_frontier_holders(engine: Any, mode: str) -> None:
    clone = (engine.clone_for_agent() if mode == "clone" else
             copy.deepcopy(engine) if mode == "deepcopy" else copy.copy(engine))
    try:
        assert clone._frontier is not engine._frontier
    finally:
        if mode != "shallow":
            clone.shutdown()


def test_holder_has_only_five_fields_and_properties_forward(engine: Any) -> None:
    from dataclasses import fields

    from hermes_lcm.frontier_state import FrontierState

    mapping = {
        "_ingest_cursor": "cursor",
        "_ingest_cursor_needs_reconcile": "needs_reconcile",
        "_last_compacted_store_id": "compacted_store_id",
        "_compress_commit_proof": "commit_proof",
        "_last_emission_descriptors": "emission_descriptors",
    }
    assert {field.name for field in fields(FrontierState)} == set(mapping.values())
    assert isinstance(engine._frontier, FrontierState)
    state = FrontierState()
    for legacy, field in mapping.items():
        assert getattr(state, field) is DEFAULTS[legacy]
        value: Any = {"identity": []} if "proof" in field or "descriptors" in field else 1234
        setattr(engine, legacy, value)
        assert getattr(engine._frontier, field) is value
        setattr(engine._frontier, field, DEFAULTS[legacy])
        assert getattr(engine, legacy) is DEFAULTS[legacy]


def test_holder_exists_before_first_legacy_init_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    holders: list[Any] = []

    def record(instance: Any, name: str, value: Any) -> None:
        if name in DEFAULTS:
            holders.append(instance._frontier)
        object.__setattr__(instance, name, value)

    monkeypatch.setattr(LCMEngine, "__setattr__", record)
    instance = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        assert len(holders) == 5
        assert all(holder is instance._frontier for holder in holders)
    finally:
        instance.shutdown()


@pytest.mark.parametrize("factory", [LCMEngine.__new__, object.__new__])
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("site", ["progress", "runtime", "invalidate", "adopt"])
def test_frontier_transition_sites_keep_write_order_and_lazy_creation(
    factory: Any, seeded: bool, site: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from hermes_lcm.frontier_state import FrontierState
    from hermes_lcm.survival_fit import FitPlan

    instance = factory(LCMEngine)
    if seeded:
        instance._frontier = FrontierState(17, True, 23, {"proof": []}, {"rows": []})
    previous = vars(instance._frontier).copy() if seeded else vars(FrontierState()).copy()
    writes: list[tuple[str, Any]] = []

    def record(holder: Any, name: str, value: Any) -> None:
        writes.append((name, value))
        object.__setattr__(holder, name, value)

    monkeypatch.setattr(FrontierState, "__setattr__", record)
    expected = [] if seeded else list(previous.items())
    if site in {"progress", "runtime"}:
        expected += [("compacted_store_id", 0), ("cursor", 0), ("needs_reconcile", False)]
        if site == "progress":
            instance._reset_compaction_progress()
        else:
            instance._compaction_request = SimpleNamespace()
            instance._pressure_yield_reset_epoch = 0
            instance._reset_session_scoped_runtime_state()
            expected += [("commit_proof", None), ("emission_descriptors", None)]
        assert instance._last_ingest_reconciliation == {"action": "none", "reason": "not run"}
    else:
        fitted = [{"role": "user", "content": "kept"}]
        instance._survival_measure = lambda rows: len(rows)
        instance._survival_record = lambda *args, **kwargs: None
        plan = FitPlan(fitted, 1, [1], False, "", False, [])
        assert instance._apply_survival_fit(
            fitted * 2, plan, 2, 1, "probe", after_exception=False,
            persisted=site == "adopt", exit_fit=False,
        ) is fitted
        expected += [("cursor", 1)] if site == "adopt" else [("cursor", 0), ("needs_reconcile", True)]
    assert writes == expected
    final = previous | dict(expected)
    assert vars(instance._frontier) == final
    for name in ("commit_proof", "emission_descriptors"):
        assert getattr(instance._frontier, name) is final[name]


def test_constructor_creates_one_frontier_without_transition_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness import engine_context

    holders: list[Any] = []

    def record(instance: Any, name: str, value: Any) -> None:
        if name == "_frontier":
            holders.append(value)
        object.__setattr__(instance, name, value)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("constructor reached a reset or survival-fit site")

    monkeypatch.setattr(LCMEngine, "__setattr__", record)
    for name in ("_reset_compaction_progress", "_reset_session_scoped_runtime_state", "_apply_survival_fit"):
        monkeypatch.setattr(LCMEngine, name, unexpected)
    with engine_context(tmp_path) as instance:
        assert len(holders) == 1
        assert holders[0] is instance._frontier


@pytest.fixture
def transition_state(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], list[str]]:
    from hermes_lcm.frontier_state import FrontierState

    state = FrontierState(17, True, 23, {"proof": []}, {"rows": []})
    before = vars(state).copy()
    writes: list[str] = []

    def record(holder: Any, name: str, value: Any) -> None:
        writes.append(name)
        object.__setattr__(holder, name, value)

    monkeypatch.setattr(FrontierState, "__setattr__", record)
    return state, before, writes


def assert_frontier_transition(
    observation: tuple[Any, dict[str, Any], list[str]], updates: dict[str, Any],
) -> None:
    state, before, writes = observation
    assert writes == list(updates)
    assert vars(state) == before | updates
    for name in before.keys() - updates.keys():
        assert getattr(state, name) is before[name]


def test_invalidate_changes_only_cursor_and_reconcile(transition_state: Any) -> None:
    transition_state[0].invalidate()
    assert_frontier_transition(transition_state, {"cursor": 0, "needs_reconcile": True})


def test_invalidate_drops_proof_before_cursor_and_reconcile(transition_state: Any) -> None:
    transition_state[0].invalidate(drop_proof=True)
    assert_frontier_transition(transition_state, {"commit_proof": None, "cursor": 0, "needs_reconcile": True})


def test_adopt_changes_only_cursor(transition_state: Any) -> None:
    transition_state[0].adopt(31)
    assert_frontier_transition(transition_state, {"cursor": 31})


def test_reset_progress_keeps_proofs(transition_state: Any) -> None:
    transition_state[0].reset_progress()
    assert_frontier_transition(transition_state, {"compacted_store_id": 0, "cursor": 0, "needs_reconcile": False})


def test_reset_proofs_keeps_progress(transition_state: Any) -> None:
    transition_state[0].reset_proofs()
    assert_frontier_transition(transition_state, {"commit_proof": None, "emission_descriptors": None})


@pytest.mark.parametrize("method, args, updates", [
    ("adopt_trusted", (31,), {"cursor": 31, "needs_reconcile": False}),
    ("restart_cursor", (), {"cursor": 0, "compacted_store_id": 0}),
    ("bind_frontier", (31,), {"compacted_store_id": 31}),
    ("publish_frontier", (31,), {"compacted_store_id": 31}),
])
def test_boundary_transition_fields_and_order(
    transition_state: Any, method: str, args: tuple[Any, ...], updates: dict[str, Any],
) -> None:
    getattr(transition_state[0], method)(*args)
    assert_frontier_transition(transition_state, updates)


@pytest.mark.parametrize("proof", [False, True])
@pytest.mark.parametrize("descriptors", [False, True])
def test_binding_filter_subsets(transition_state: Any, proof: bool, descriptors: bool) -> None:
    transition_state[0].filter_by_binding(proof=proof, descriptors=descriptors)
    updates = {}
    if proof:
        updates["commit_proof"] = None
    if descriptors:
        updates["emission_descriptors"] = None
    assert_frontier_transition(transition_state, updates)


@pytest.mark.parametrize("field, key, value", [
    ("commit_proof", "carry_ranges", [("source", 1, 7)]),
    ("commit_proof", "session_id", "child"),
    ("emission_descriptors", "session_id", "child"),
    ("commit_proof", "input", None),
])
def test_child_transfer_keeps_dict_identity(
    transition_state: Any, field: str, key: str, value: Any,
) -> None:
    state, before, _ = transition_state
    target = before[field]
    previous = target.copy()
    state.transfer_to_child(target, key, value)
    assert target == previous | {key: value}
    assert target[key] is value
    assert_frontier_transition(transition_state, {})


def test_child_emission_scope_keeps_enclosing_dict(transition_state: Any) -> None:
    state, before, _ = transition_state
    emission = {"scope": {"session_id": "source", "reset_epoch": 7}}
    before["commit_proof"]["emissions"] = [emission]
    scope = {**emission["scope"], "session_id": "child"}
    state.transfer_to_child(emission, "scope", scope)
    assert before["commit_proof"]["emissions"][0] is emission
    assert emission["scope"] is scope
    assert_frontier_transition(transition_state, {})


def test_end_consumed_keeps_captured_dict(transition_state: Any) -> None:
    state, before, _ = transition_state
    proof = before["commit_proof"]
    previous = proof.copy()
    state.mark_end_consumed(proof)
    assert proof == previous | {"end_consumed": True}
    assert_frontier_transition(transition_state, {})
