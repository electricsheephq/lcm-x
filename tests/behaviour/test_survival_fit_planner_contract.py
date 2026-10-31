"""Pin planner decisions and ordered effects before the planner extraction (I1/I2/I3/I10)."""

from collections.abc import Callable
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import pytest

harness = import_module("tests.harness")
fit = import_module("hermes_lcm.survival_fit")
Row = dict[str, Any]


@pytest.mark.parametrize("path", ["prefix", "emergency", "no-prefix", "no-cut", "exit", "exit-skip"])
@pytest.mark.parametrize("state", ["persisted", "short-cursor", "reconcile", "exception"])
@pytest.mark.parametrize("caps", ["none", "window", "request"])
def test_planner_fields_and_effect_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, state: str, caps: str,
    record_property: Callable[[str, object], None],
) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path) as engine:
        rows: list[Row] = [{"role": "system", "content": "system"},
                           {"role": "user", "content": "summary"},
                           {"role": "user", "content": "old"},
                           {"role": "user", "content": "new"}]
        fitted = [rows[0], rows[-1]]
        prefix = 1 if path == "no-prefix" else 2
        reason = "exit_fit:contract" if path.startswith("exit") else "contract"
        persisted = state == "persisted"
        engine._session_id = "contract"
        engine.context_length = 120
        engine._ingest_cursor = 2 if state == "short-cursor" else len(rows)
        engine._ingest_cursor_needs_reconcile = state == "reconcile"
        engine._bypasses_lcm_context_management = lambda: False
        engine._survival_fit_budget = lambda *a: 110 if len(a) == 2 else 70
        engine._survival_measure = lambda value: 100
        engine._is_verified_replay_scaffold_message = lambda row: prefix == 2 and row is rows[1]
        engine._generated_context_carrier_remainder = lambda row: None
        events: list[Any] = []
        cls = type(engine)
        read: Any = cls.__getattribute__
        write: Any = cls.__setattr__

        def get(obj: Any, name: str) -> Any:
            if obj is engine and name in ("_ingest_cursor", "_ingest_cursor_needs_reconcile"):
                events.append(["read", name])
            return read(obj, name)

        def put(obj: Any, name: str, value: Any) -> None:
            if obj is engine and name == "_survival_cut_carrier":
                events.append(["carrier", value])
            write(obj, name, value)

        monkeypatch.setattr(cls, "__getattribute__", get)
        monkeypatch.setattr(cls, "__setattr__", put)
        mapping = {id(row): i + 10 for i, row in enumerate(rows[1:])}

        def store(value: list[Row]) -> dict[int, int]:
            assert all(a is b for a, b in zip(value, rows[1:], strict=True))
            events.append(["store", value])
            return mapping

        def overhead(*args: Any) -> int:
            assert args == (rows, 17)
            events.append(["overhead"])
            return 20

        def tail(value: list[Row]) -> int:
            assert value is rows
            events.append(["tail"])
            return 3

        def cut(value: list[Row], lead: int, budget: int, durable: bool, why: str,
                ids: dict[int, int], whole: bool, **kw: Any) -> Any:
            assert value is rows and ids is mapping
            events.append(["cut", lead, budget, durable, why, whole, kw])
            engine._survival_cut_carrier = None
            if whole:
                engine._survival_cut_carrier = "prefix-carrier"
            if path in ("no-cut", "exit-skip") or (path == "emergency" and whole):
                return None
            return fitted, 2, [10, 11], True, "notice"

        def apply(value: list[Row], plan: Any, before: int, budget: int, why: str, **kw: Any) -> list[Row]:
            assert value is rows and plan.fitted is fitted
            assert all(a is b for a, b in zip(plan.emergency_span, rows[1:prefix], strict=True))
            events.append(["apply", vars(plan), before, budget, why, kw, engine._survival_cut_carrier])
            return rows  # stop at the caller-owned apply boundary; stamping is covered separately

        engine._get_store_id_map_for_messages = store
        engine._survival_host_overhead = overhead
        engine._fresh_tail_start = tail
        engine._survival_cut = cut
        engine._apply_survival_fit = apply
        monkeypatch.setattr(fit.logger, "info", lambda *args: events.append(["info", *args]))
        kwargs = {f"{caps}_cap": 90} if caps != "none" else {}
        assert engine._survival_fit(rows, rows, 17, reason, after_exception=state == "exception",
                                    summary_failure="failure", fit_cause="cause", **kwargs) is rows
        expected: list[Any] = []
        if state != "exception":
            expected.append(["read", "_ingest_cursor_needs_reconcile"])
            if state != "reconcile":
                expected.append(["read", "_ingest_cursor"])
        expected.append(["store", rows[1:]])
        if path.startswith("exit"):
            expected.append(["tail"])
        if prefix == 2:
            expected.extend([["cut", 2, 70, persisted, reason, True,
                              {"keep_from": 3} if path.startswith("exit") else {}],
                             ["carrier", None], ["carrier", "prefix-carrier"]])
        if not path.startswith("exit"):
            if caps == "none":
                expected.append(["overhead"])
            if path in ("emergency", "no-prefix", "no-cut"):
                expected.extend([["cut", 1, 70, persisted, reason, False,
                                  {"hard_cap": 100 if caps == "none" else None}], ["carrier", None]])
        if path == "exit-skip":
            expected.append(["info", "LCM exit fit skipped: no whole-turn cut between the summary prefix and the fresh tail "
                             "reaches the exit cap (tokens=%d, budget=%d)", 100, 70])
        elif path != "no-cut":
            expected.append(["apply", dict(fitted=fitted, count=2, ids=[10, 11], projected=True, notice="notice",
                                           emergency=path == "emergency", emergency_span=rows[1:prefix]),
                             100, 70, reason, dict(after_exception=state == "exception", persisted=persisted,
                                                  exit_fit=path == "exit", summary_failure="failure", fit_cause="cause"),
                             "prefix-carrier" if path in ("prefix", "exit") else None])
        assert events == expected
        assert engine._survival_cut_carrier == ("prefix-carrier" if path in ("prefix", "exit", "exit-skip") else None)
        record_property("planner_trace", json.dumps(events, sort_keys=True))


def test_real_cut_reforms_carrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                record_property: Callable[[str, object], None]) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path) as engine:
        rows: list[Row] = [{"role": "user", "content": "summary\n\nold"},
                           {"role": "user", "content": "kept"}, {"role": "user", "content": "new"}]
        engine._session_id = "contract"
        engine._ingest_cursor = len(rows)
        engine._ingest_cursor_needs_reconcile = False
        engine._bypasses_lcm_context_management = lambda: False
        engine._survival_fit_budget = lambda *a: 70
        engine._survival_measure = lambda value: 100 if value is rows else 50
        engine._is_verified_replay_scaffold_message = lambda row: False
        engine._generated_context_carrier_remainder = lambda row: (
            row["content"][9:] if row["content"].startswith("summary\n\n") else None)
        engine._verified_lcm_summary_prefix_end = lambda text: 7
        engine._survival_generated = lambda row: False
        engine._survival_host_overhead = lambda *a: 0
        engine._get_store_id_map_for_messages = lambda value: {id(row): i + 10 for i, row in enumerate(rows)}
        # Identity emission is independently tested; keep this carrier recipe deterministic.
        monkeypatch.setattr(fit, "identity_emit_enabled", lambda: False)
        plans: list[Any] = []

        def apply(value: list[Row], plan: Any, *args: Any, **kw: Any) -> list[Row]:
            plans.append(plan)
            assert kw["persisted"] is True
            return rows

        engine._apply_survival_fit = apply
        assert engine._survival_fit(rows, rows, 0, "contract") is rows
        plan, = plans
        merged = {"role": "user", "content": "summary\n\nkept"}
        assert vars(plan) == dict(fitted=[merged, rows[-1]], count=1, ids=[10], projected=False,
                                 notice=fit._NOTICE.format(n=1, first=10, last=10), emergency=False,
                                 emergency_span=rows[:1])
        assert engine._survival_cut_carrier == (merged, "summary\n\n", 11)
        assert engine._survival_cut_carrier[0] is plan.fitted[0]
        record_property("planner_trace", json.dumps([vars(plan), engine._survival_cut_carrier], sort_keys=True))
