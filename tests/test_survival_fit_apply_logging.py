"""Pin observable apply order with the real survival recorder and measurement."""

import logging
from importlib import import_module
from typing import Any
from unittest.mock import Mock

import pytest

from tests.test_survival_fit_apply_contract import fit_engine as fit_engine

fit = import_module("hermes_lcm.survival_fit")
tokens = import_module("hermes_lcm.tokens")
fixtures = import_module("tests.test_survival_fit_apply_contract")
Row = dict[str, Any]


@pytest.mark.parametrize("path,reason,before,after", [
    ("best-effort", "contract", 322, 117),
    ("best-effort", "exit_fit:contract", 322, 117),
    ("no-shrink", "contract", 222, 222),
    ("emergency-prefix", "contract", 222, 11),
])
@pytest.mark.parametrize("after_exception", [False, True], ids=["persisted", "exception"])
@pytest.mark.parametrize("counter_failure", [False, True], ids=["counter-ok", "counter-failed"])
def test_apply_logging_measurement_and_cursor_order(
    fit_engine: Any, path: str, reason: str, before: int, after: int, after_exception: bool,
    counter_failure: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    emergency = path != "best-effort"
    engine, rows, fitted = fixtures.prepared_fit(
        fit_engine, emergency=emergency, prefix="dropped" if emergency else "same")
    rows[2]["content"] = "old " * 200
    if path == "best-effort":
        rows[-1]["content"] = "new " * 100
    elif path == "no-shrink":
        fitted[:] = rows

    # Use the real measurement, with the deterministic offline token fallback.
    monkeypatch.setattr(fit, "_host_estimate", lambda _messages: None)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(tokens, "_count_tokens_cached", lambda text, _generation: tokens._count_tokens_core(text))
    measurements: list[list[int]] = []

    def measure(messages: list[Row]) -> int:
        measurements.append([id(row) for row in messages])
        return int(fit.SurvivalFitMixin._survival_measure(engine, messages))

    engine._survival_measure = measure
    engine._survival_record = fit.SurvivalFitMixin._survival_record.__get__(engine)
    engine._survival_fit_budget = lambda _m, _o, _w=None, cap=None: 1000 if cap is None else cap
    engine._conversation_id = ""
    engine._survival_fit_warned = set()
    engine._store_complete_node_covered = lambda ids: set(ids)
    monkeypatch.setattr(engine._store, "update_metadata_json", Mock())
    if counter_failure:
        # No-shrink has no "applied" log. This real recorder warning makes its
        # ordering observable even on that path, without mocking _survival_record.
        engine._store.update_metadata_json.side_effect = RuntimeError("counter unavailable")

    warning_cursors: list[tuple[int, bool]] = []
    emit = caplog.handler.emit

    def capture(record: logging.LogRecord) -> None:
        if record.name == "hermes_lcm.survival_fit" and record.getMessage().startswith(
                "LCM survival fit dropped the summary prefix"):
            warning_cursors.append((engine._ingest_cursor, engine._ingest_cursor_needs_reconcile))
        emit(record)

    monkeypatch.setattr(caplog.handler, "emit", capture)
    with caplog.at_level(logging.INFO, logger="hermes_lcm.survival_fit"):
        result = engine._survival_fit(rows, rows, 0, reason, after_exception=after_exception, request_cap=70)

    assert result is (rows if path == "no-shrink" else fitted)
    expected_measurements = [[id(row) for row in rows], [id(row) for row in fitted]]
    expected_logs: list[tuple[str, int, str]] = []

    def warning(message: str) -> None:
        expected_logs.append(("hermes_lcm.survival_fit", logging.WARNING, message))

    if path == "no-shrink":
        warning(f"LCM survival fit could not shorten the list (before={before}, budget=70, reason={reason})")
    else:
        if path == "emergency-prefix":
            expected_measurements.append([id(rows[1])])
            warning("LCM survival fit dropped the summary prefix (emergency: prefix=6 tokens, budget=70)")
        if path == "best-effort":
            warning(f"LCM survival fit could not reach budget (after={after}, budget=70, reason={reason})")
        expected_logs.append((
            "hermes_lcm.survival_fit", logging.INFO if reason.startswith("exit_fit:") else logging.WARNING,
            f"LCM survival fit applied (reason={reason}, conversation=contract, dropped_rows=2, "
            f"store_ids=11..12, uncovered_rows=0, projected=True, tokens={before}->{after}, budget=70)",
        ))
    if counter_failure:
        warning(f"LCM survival-fit counter write failed (projected={path != 'no-shrink'}); "
                "/lcm doctor under-counts survival fits for this store")
    assert [(record.name, record.levelno, record.getMessage()) for record in caplog.records] == expected_logs
    assert measurements == expected_measurements
    assert warning_cursors == ([(len(rows), False)] if path == "emergency-prefix" else [])
    assert (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile) == (
        (len(rows), False) if path == "no-shrink" else (0, True) if after_exception else (len(fitted), False))
