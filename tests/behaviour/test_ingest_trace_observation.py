"""Product mutations that must be visible to the ingest trace observers."""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour import ingest_trace_harness as harness


def trace(name: str, root: Path) -> list[str]:
    case = next(case for case in harness.SCENARIOS if case.name == name)
    return harness.observe(case, root)


def baseline(name: str, root: Path) -> list[str]:
    expected: list[str] = json.loads(harness.GOLDEN.read_text())[name]
    actual = trace(name, root)
    assert actual == expected, harness.first_difference(expected, actual)
    return actual


def test_rotation_dag_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "sequence-rotation"
    expected = baseline(name, tmp_path / "baseline")
    dag: Any = importlib.import_module("hermes_lcm.dag")
    original = dag.SummaryDAG.get_session_nodes

    def phantom_child(self: Any, session_id: str, *args: Any, **kwargs: Any) -> Any:
        nodes = original(self, session_id, *args, **kwargs)
        if session_id == "child":
            return [*nodes, dag.SummaryNode(session_id="child", summary="mutation-only node")]
        return nodes

    monkeypatch.setattr(dag.SummaryDAG, "get_session_nodes", phantom_child)
    actual = trace(name, tmp_path / "mutated")
    assert actual != expected, "G1 missed: child DAG mutation left the golden unchanged"


@pytest.mark.parametrize("mode", ["ignored", "stateless"])
def test_cold_admission_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    name = "admission-" + mode
    expected = baseline(name, tmp_path / "baseline")
    original = harness.engine_module.LCMEngine.on_session_start
    starts = 0

    def lose_cold_filter(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal starts
        result = original(self, *args, **kwargs)
        starts += 1
        if starts == 2:
            setattr(self, "_session_" + mode, False)
        return result

    monkeypatch.setattr(harness.engine_module.LCMEngine, "on_session_start", lose_cold_filter)
    actual = trace(name, tmp_path / "mutated")
    assert starts == 2
    assert actual != expected, "G2 missed: lost cold admission filter left the golden unchanged"


@pytest.mark.parametrize("ownership", ["other", "absent", "none"])
def test_pending_tail_store_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ownership: str,
) -> None:
    name = "sqlite-locked"
    expected = baseline(name, tmp_path / "baseline")
    engine_type = harness.engine_module.LCMEngine
    original = engine_type._ingest_messages
    retry = engine_type._retry_pending_session_tails
    changed: list[tuple[dict[str, Any], Any]] = []
    mutations = 0

    def restore() -> None:
        for tail, store in changed:
            tail["store"] = store
        changed.clear()

    def lose_owner(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal mutations
        restore()
        try:
            return original(self, *args, **kwargs)
        finally:
            for tail in self._pending_session_tails:
                changed.append((tail, tail["store"]))
                if ownership == "absent":
                    del tail["store"]
                else:
                    tail["store"] = None if ownership == "none" else object()
                mutations += 1

    def retry_restored(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Restore before any consumer runs: this probe isolates snapshot
        # identity from retry failures, which the existing trace already sees.
        restore()
        return retry(self, *args, **kwargs)

    monkeypatch.setattr(engine_type, "_ingest_messages", lose_owner)
    monkeypatch.setattr(engine_type, "_retry_pending_session_tails", retry_restored)
    actual = trace(name, tmp_path / "mutated")
    assert mutations > 0
    assert actual != expected, "G3 missed: changed pending-tail owner left the golden unchanged"


@pytest.mark.parametrize("phase", ["next", "cold"])
@pytest.mark.parametrize("observer", ["store", "log", "_store", "_dag", "_lifecycle", "_host_rewrite_store"])
def test_followup_observer_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, observer: str,
) -> None:
    name = "admission-full"
    expected = baseline(name, tmp_path / "baseline")
    original = harness.engine_module.LCMEngine._ingest_messages
    calls = 0
    mutations = 0

    def extra_read(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls, mutations
        calls += 1
        result = original(self, *args, **kwargs)
        if calls == (2 if phase == "next" else 3):
            if observer == "store":
                self._store.get_session_count(self._session_id)
            elif observer == "log":
                logging.getLogger("hermes_lcm.engine").warning("mutation-only follow-up log")
            else:
                getattr(self, observer)._conn.execute("SELECT 'mutation-only follow-up SQL'").fetchone()
            mutations += 1
        return result

    monkeypatch.setattr(harness.engine_module.LCMEngine, "_ingest_messages", extra_read)
    actual = trace(name, tmp_path / "mutated")
    assert calls == 3 and mutations == 1
    assert actual != expected, f"G4 missed: {phase} {observer} left the golden unchanged"
