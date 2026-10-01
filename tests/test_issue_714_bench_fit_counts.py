"""#714: process cells count exit fits like in-process cells; B8 sees a fit that could not shorten the list."""
from __future__ import annotations

from types import SimpleNamespace

import bench.instruments.reliability.process_cell as PC
from tests.test_reliability_scorers import clean_events, clean_rows, make, probe

EXIT = ("LCM survival fit applied (reason=exit_fit:compressed, conversation=c, dropped_rows=4, store_ids=1..4, "
        "before=900, after=700, budget=760)")
UNSHORTENED = "LCM survival fit could not shorten the list (before=900, budget=500, reason={})"


def test_unshortened_non_exit_fit_is_counted_and_fails_b8(tmp_path):
    fields = probe.phase_log_fields("LCM compaction #1: done\n" + UNSHORTENED.format("compressed") + "\n")
    assert fields["log_counts"]["fit_unshortened"] == 1 and fields["log_counts"]["survival_fit"] == 0
    out = make(tmp_path, rows=clean_rows(), events=clean_events(), phase=fields)
    assert "B8" in out["failed_bars"]


def test_unshortened_exit_fit_is_not_a_miss(tmp_path):
    fields = probe.phase_log_fields("LCM compaction #1: done\n" + UNSHORTENED.format("exit_fit:compressed") + "\n")
    assert fields["log_counts"]["fit_unshortened"] == 0
    out = make(tmp_path, rows=clean_rows(), events=clean_events(), phase=fields)
    assert "B8" not in out["failed_bars"]


def test_process_cell_counts_a_routine_exit_fit_once(tmp_path, monkeypatch):
    (tmp_path / "observer.jsonl").write_text("")
    monkeypatch.setattr(PC, "cite_all", lambda src: {})
    monkeypatch.setattr(PC, "session_count", lambda home: 0)
    cell = SimpleNamespace(d=tmp_path, phase="p1", transport="acp-process", sid="s", host={"src": str(tmp_path)},
                           home=tmp_path, host_log=lambda: "LCM compaction #1: done\n" + EXIT + "\n")
    rec = PC.ProcessCell.phase_record(cell, 1, {})
    assert rec["log_counts"]["exit_fit"] == 1 and rec["log_counts"]["survival_fit"] == 0
    assert rec["log_counts"] == probe._log_counts(cell.host_log())
