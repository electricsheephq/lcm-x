"""Ruling B: preserve non-date base bytes at every occurrence-time consumer."""

import hashlib
import json
from pathlib import Path

import pytest

from hermes_lcm import requirements_compiler
from hermes_lcm.occurrence_time import resolve_occurrence_time
from tests.issue_1811_r2_cells import compile_cell, review_cells

BASE = json.loads(
    Path(__file__).with_name("issue_1811_r2_base_hashes.json").read_text()
)
CELLS = review_cells()


@pytest.mark.parametrize("cid", sorted(BASE["hashes"]))
def test_non_date_review_matrix_base_identity(tmp_path, monkeypatch, cid):
    monkeypatch.setattr(requirements_compiler.time, "perf_counter", lambda: 1.0)
    raw = compile_cell(tmp_path, CELLS[cid])
    assert hashlib.sha256(raw.encode()).hexdigest() == BASE["hashes"][cid]


def _repro(tmp_path, monkeypatch, name):
    monkeypatch.setattr(requirements_compiler.time, "perf_counter", lambda: 1.0)
    raw = compile_cell(tmp_path, BASE["repros"][name]["cell"])
    result = json.loads(raw)
    actual = [
        result["contract"]["operation"],
        result["state"],
        result["reason_code"],
        (result.get("direct_fact") or {}).get("value"),
        (result.get("computation") or {}).get("result"),
    ]
    assert actual == BASE["repros"][name]["expected"]


def test_sum_whole_row_base_grounding(tmp_path, monkeypatch):
    _repro(tmp_path, monkeypatch, "sum_whole_row")


def test_non_date_as_of_keeps_base_occurrence(tmp_path, monkeypatch):
    _repro(tmp_path, monkeypatch, "none_as_of")


def test_latest_two_candidates_keeps_base_selection(tmp_path, monkeypatch):
    _repro(tmp_path, monkeypatch, "latest_pick")


def test_carried_latest_keeps_base_occurrence(tmp_path, monkeypatch):
    _repro(tmp_path, monkeypatch, "carried_latest")


def test_carried_mixed_separator_keeps_base_predicate(tmp_path, monkeypatch):
    _repro(tmp_path, monkeypatch, "carried_mixed_separator")


@pytest.mark.parametrize(
    "quote,day",
    [
        ("We met on 2023-01-10 and yesterday.", "2023-01-10"),
        ("We met on 2023-01/10, January 12, 2023 and February 3, 2023.", "2023-01-10"),
        ("We met on January 10, 2023 and yesterday.", "2023-03-21"),
        ("We met yesterday and last week.", "2023-03-21"),
    ],
)
def test_resolver_base_mode_bypasses_new_withhold(quote, day):
    base = resolve_occurrence_time(
        quote, observed_at=123, session_date="2023-03-22", withhold_mixed=False
    )
    default = resolve_occurrence_time(quote, observed_at=123, session_date="2023-03-22")
    assert base["event_date"] == day
    assert default["event_date"] is None


def test_large_cache_returns_local_value_during_clear(monkeypatch):
    from hermes_lcm import occurrence_time

    class ClearingMap(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self.clear()

    quote = "2023-01-10 and yesterday. " + "x" * 1300
    monkeypatch.setattr(occurrence_time, "_large_quote_date_info", ClearingMap())
    assert occurrence_time._quote_date_info(
        quote
    ) == occurrence_time._parse_quote_date_info(quote)
