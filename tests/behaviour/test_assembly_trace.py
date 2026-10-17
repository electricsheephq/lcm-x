"""Frozen assembly channels; intentional rebaseline only after a declared delta."""
from __future__ import annotations

import gc
import importlib
import inspect
import json
import sys
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour.assembly_trace_harness import (
    EXTERNAL_ATTRIBUTES, GOLDEN, OBSERVED_ATTRIBUTES, SCENARIOS, TRACE_GUIDANCE, Canonical, Scenario,
    dag_module, engine_attributes, engine_module, first_difference, observe,
)


def test_assembly_attribute_inventory() -> None:
    discovered = engine_attributes(inspect.getsource(engine_module))
    assert discovered == OBSERVED_ATTRIBUTES - EXTERNAL_ATTRIBUTES, (
        f'Assembly state inventory changed: added={sorted(discovered - OBSERVED_ATTRIBUTES)}, '
        f'removed={sorted(OBSERVED_ATTRIBUTES - EXTERNAL_ATTRIBUTES - discovered)}; '
        'update OBSERVED_ATTRIBUTES/EXTERNAL_ATTRIBUTES.\n' + TRACE_GUIDANCE)


@pytest.mark.parametrize('case', SCENARIOS, ids=lambda case: case.name)
def test_assembly_golden(case: Scenario, tmp_path: Path) -> None:
    expected = json.loads(GOLDEN.read_text())[case.name]
    actual = observe(case, tmp_path.resolve())
    difference = first_difference(expected, actual)
    assert difference is None, f'{case.name}: first divergence: {difference}\n' + TRACE_GUIDANCE


def test_assembly_defers_prior_cycle_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = importlib.import_module('hermes_lcm.vector_store')
    original = vectors.VectorStore._binary_fully_synced
    prior = dag_module.SummaryDAG(tmp_path / 'prior.db')
    prior.close()
    prior.cycle = prior
    pending = [prior]
    del prior
    threshold, enabled = gc.get_threshold(), gc.isenabled()

    def pressure(store: Any, *args: Any, **kwargs: Any) -> Any:
        gc.set_threshold(1000000, 1000000, 1000000)
        pending.clear()
        gc.set_threshold(1, 1, 1)
        # Force an automatic collection opportunity inside the SQL observation.
        allocations: list[list[object]] = [[] for _ in range(200)]
        assert len(allocations) == 200
        gc.set_threshold(*threshold)
        return original(store, *args, **kwargs)

    monkeypatch.setattr(vectors.VectorStore, '_binary_fully_synced', pressure)
    case = next(case for case in SCENARIOS if case.name == 'recall-on')
    try:
        actual = observe(case, tmp_path.resolve())
    finally:
        gc.set_threshold(*threshold)
    assert gc.isenabled() == enabled
    difference = first_difference(json.loads(GOLDEN.read_text())[case.name], actual)
    assert difference is None, f'{case.name}: first divergence: {difference}\n' + TRACE_GUIDANCE


def test_fixture_scenario_inventory() -> None:
    names = [case.name for case in SCENARIOS]
    assert len(names) == len(set(names)), "Scenario names must be unique.\n" + TRACE_GUIDANCE
    assert set(json.loads(GOLDEN.read_text())) == set(names), (
        "Scenario inventory differs; update SCENARIOS and the fixture scenario keys together.\n" + TRACE_GUIDANCE)



@pytest.mark.parametrize("equals", [False, True], ids=["separate", "equals"])
def test_repo_option_binds_selected_tree(tmp_path: Path, equals: bool) -> None:
    root = Path(__file__).resolve().parents[2]
    selected = tmp_path / "selected"
    selected.mkdir()
    for path in root.iterdir():
        if path.suffix == ".py" or (path.is_dir() and not path.name.startswith(".")):
            (selected / path.name).symlink_to(path, target_is_directory=path.is_dir())
    option = [f"--repo={selected}"] if equals else ["--repo", str(selected)]
    script = """
import runpy, sys
try:
    runpy.run_module("tests.behaviour.assembly_trace_harness", run_name="__main__")
except SystemExit as exc:
    if exc.code:
        raise
print("Product package:", sys.modules["hermes_lcm"].__path__[0])
"""
    result = subprocess.run([sys.executable, "-c", script, *option, "--help"],
                            cwd=root, capture_output=True, text=True, check=True)
    assert f"Product package: {selected.resolve()}" in result.stdout


@pytest.mark.parametrize("option", [["--repo"], ["--repo", "--out"], ["--repo="]])
def test_repo_option_rejects_missing_value(option: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", "tests.behaviour.assembly_trace_harness", *option],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 2
    assert "argument --repo: expected" in result.stderr


def test_store_associations_use_message_aliases(tmp_path: Path) -> None:
    first, second = {'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'second'}
    canonical = Canonical(tmp_path)
    canonical.graph([first, second])
    expected = canonical.value({id(first): 11, id(second): 22}, 'store_ids_by_message_id')
    swapped = canonical.value({id(second): 11, id(first): 22}, 'store_ids_by_message_id')
    assert first_difference([json.dumps(expected, sort_keys=True)], [json.dumps(swapped, sort_keys=True)]) is not None
    assert expected == {canonical.alias(first): 11, canonical.alias(second): 22}
    assert expected == canonical.value({id(second): 22, id(first): 11}, 'store_ids_by_message_id')
