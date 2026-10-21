"""Characterize the survival fit's cursor, warning and emission side effects."""

from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tests.harness import engine_context

# The test bootstrap registers the plugin package at runtime.
fit = import_module("hermes_lcm.survival_fit")
Row = dict[str, Any]


@pytest.fixture
def fit_engine(tmp_path: Path) -> Iterator[Any]:
    with engine_context(tmp_path) as engine:
        yield engine


def prepared_fit(engine: Any, *, state: str = "persisted", after: int = 60, carrier: bool = False,
                 emergency: bool = False, prefix: str = "same") -> tuple[Any, list[Row], list[Row]]:
    rows: list[Row] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "summary", "prefix": True},
        {"role": "user", "content": "old"},
        {"role": "user", "content": "new"},
    ]
    fitted = [rows[0], rows[-1]]
    if prefix != "dropped":
        fitted.insert(1, rows[1] if prefix == "same" else dict(rows[1]))
    engine._session_id = "contract"
    engine._ingest_cursor = len(rows) if state != "unpersisted" else 1
    engine._ingest_cursor_needs_reconcile = state == "reconcile"
    engine._bypasses_lcm_context_management = lambda: False
    engine._survival_fit_budget = lambda _m, _o, _w=None, cap=None: 200 if cap is None else cap
    engine._survival_measure = lambda value: 100 if value is rows else after if value is fitted else 17
    engine._is_verified_replay_scaffold_message = lambda row: row is rows[1]
    engine._generated_context_carrier_remainder = lambda row: None
    engine._get_store_id_map_for_messages = lambda value: {}
    engine._fresh_tail_start = lambda value: 3
    engine._survival_cut = Mock(side_effect=(
        [None, (fitted, 2, [11, 12], True, "notice")] if emergency
        else [(fitted, 2, [11, 12], True, "notice")]))
    merged = {"role": "user", "content": "summary + new"}
    engine._survival_cut_carrier = (merged, (4, 8), 13) if carrier else None
    engine._survival_fit_emission = {"previous": True}
    engine._survival_record = Mock()
    return engine, rows, fitted


@pytest.mark.parametrize("state", ["persisted", "unpersisted", "reconcile", "exception"])
@pytest.mark.parametrize("exit_fit", [False, True], ids=["survival", "exit"])
@pytest.mark.parametrize("carrier", [False, True], ids=["plain", "carrier"])
@pytest.mark.parametrize("after", [60, 80], ids=["within-budget", "best-effort"])
def test_apply_preserves_cursor_record_and_emission(
    fit_engine: Any, state: str, exit_fit: bool, carrier: bool, after: int, caplog: pytest.LogCaptureFixture,
) -> None:
    engine, rows, fitted = prepared_fit(fit_engine, state=state, after=after, carrier=carrier)
    reason = "exit_fit:contract" if exit_fit else "contract"
    needs_reconcile = state != "persisted"
    snapshots: list[tuple[int, bool, dict[str, Any]]] = []
    engine._survival_record.side_effect = lambda *a, **kw: snapshots.append(
        (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile, engine._survival_fit_emission))
    result = engine._survival_fit(rows, rows, 0, reason, after_exception=state == "exception", request_cap=70)
    assert result is fitted
    assert snapshots == [(0 if needs_reconcile else len(fitted), needs_reconcile, {"previous": True})]
    engine._survival_record.assert_called_once_with(
        reason, 2, [11, 12], 100, after, 70, True, "notice", warn_user=not exit_fit)
    assert ("could not reach budget" in caplog.text) == (after > 70)
    assert "dropped the summary prefix" not in caplog.text
    if carrier:
        merged, span, store_id = engine._survival_cut_carrier
        assert engine._survival_fit_emission == {
            "kind": "carrier", "span": span, "retained_source": {"store_id": store_id},
            "full_identity": fit._emission_identity(merged), "row": merged}
        assert engine._survival_fit_emission["row"] is merged
    else:
        assert engine._survival_fit_emission == {"previous": True}


@pytest.mark.parametrize("after", [100, 101], ids=["equal", "larger"])
def test_no_shrink_records_failure_and_keeps_state(fit_engine: Any, after: int, caplog: pytest.LogCaptureFixture) -> None:
    engine, rows, _ = prepared_fit(fit_engine, after=after, carrier=True, emergency=True, prefix="dropped")
    result = engine._survival_fit(rows, rows, 0, "contract", after_exception=True, request_cap=70)
    assert result is rows
    assert (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile) == (len(rows), False)
    assert engine._survival_fit_emission == {"previous": True}
    engine._survival_record.assert_called_once_with(
        "contract", 0, [], 100, 100, 70, False, "", shortened=False)
    assert [record.getMessage() for record in caplog.records] == [
        "LCM survival fit could not shorten the list (before=100, budget=70, reason=contract)"]


@pytest.mark.parametrize("prefix", ["same", "equal-copy", "dropped"])
@pytest.mark.parametrize("emergency", [False, True])
def test_emergency_warning_checks_prefix_identity(
    fit_engine: Any, prefix: str, emergency: bool, caplog: pytest.LogCaptureFixture,
) -> None:
    engine, rows, fitted = prepared_fit(fit_engine, emergency=emergency, prefix=prefix)
    assert engine._survival_fit(rows, rows, 0, "contract", request_cap=70) is fitted
    expected = emergency and prefix != "same"
    assert ("dropped the summary prefix" in caplog.text) == expected
    if expected:
        assert caplog.records[0].getMessage() == (
            "LCM survival fit dropped the summary prefix (emergency: prefix=17 tokens, budget=70)")


def test_missing_cut_leaves_apply_state_untouched(fit_engine: Any) -> None:
    engine, rows, _ = prepared_fit(fit_engine, carrier=True)
    engine._survival_cut = Mock(return_value=None)
    assert engine._survival_fit(rows, rows, 0, "contract", request_cap=70) is rows
    assert (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile) == (len(rows), False)
    assert engine._survival_fit_emission == {"previous": True}
    engine._survival_record.assert_not_called()
