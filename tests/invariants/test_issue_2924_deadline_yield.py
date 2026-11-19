"""#2924 Step A: deadline yields preserve committed ingest (I1, I2, I11)."""

import hashlib
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from tests.harness import engine_context, pin_char_counter, stored_rows
from tests.test_issue_2863_host_contract import Host, ModelClock
from tests.test_issue_581_582_store_complete_survival import _turn


@pytest.fixture
def engine(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: (
        "Earlier turns.\nExpand for details about: synthetic turns", 1))
    with engine_context(
        tmp_path, window=200_000, conversation_id="C", context_threshold=0.001,
        fresh_tail_count=2, leaf_chunk_tokens=400, summary_model="probe",
        embeddings_enabled=False, threshold_full_sweep_enabled=True,
    ) as eng:
        yield eng


@pytest.fixture
def view():
    return [row for i in range(8) for row in _turn(f"T{i}", 100.0 + i, tool=i == 0)]


def test_t1_slow_ingest_yields_after_commit(engine, view, monkeypatch, caplog):
    engine._config.foreground_hard_seconds = 2
    view[0]["content"] += "\ud800"  # The canonical view is a different list.
    ingest = engine._ingest_messages
    finished = []

    def slow(*args, **kwargs):
        time.sleep(3)
        result = ingest(*args, **kwargs)
        finished.append(time.monotonic())
        return result

    monkeypatch.setattr(engine, "_ingest_messages", slow)
    sweep = Mock(wraps=engine._prepare_compression_sweep)
    monkeypatch.setattr(engine, "_prepare_compression_sweep", sweep)
    before = deepcopy(view)
    with caplog.at_level("INFO", logger="hermes_lcm.compaction"):
        result = engine.compress(view, current_tokens=8000)
    after = time.monotonic()
    assert engine.last_compression_status == "deadline_yield"
    assert result is view and view == before
    sweep.assert_not_called()
    assert 0 <= after - finished[0] < 1
    assert engine._ingest_cursor == len(view)
    assert engine._foreground_budget is None
    assert sum("check=after_ingest" in line for line in caplog.messages) == 1


def _late_ingest(engine, monkeypatch):
    clock = ModelClock(monkeypatch)
    ingest = engine._ingest_messages

    def slow(*args, **kwargs):
        result = ingest(*args, **kwargs)
        clock.offset += 121
        return result

    monkeypatch.setattr(engine, "_ingest_messages", slow)
    return clock


def _digests(rows):
    return [(row["role"], hashlib.sha256((row["content"] or "").encode()).hexdigest()) for row in rows]


def test_t3_next_pass_keeps_each_row_once_and_publishes(engine, view, monkeypatch):
    with monkeypatch.context() as patch:
        _late_ingest(engine, patch)
        assert engine.compress(view, current_tokens=8000) is view
        assert engine.last_compression_status == "deadline_yield"
    assert engine._ingest_cursor == len(view)
    assert _digests(stored_rows(engine)) == _digests(view)
    view = [*view, *_turn("next", 110.0)]
    result = engine.compress(view, current_tokens=8000)
    assert _digests(stored_rows(engine)) == _digests(view)
    assert engine._ingest_cursor == len(result)
    assert result != view and engine._last_compress_leaves[1] > 0


def test_t4_three_yields_leave_failure_and_hold_state_clear(engine, view, monkeypatch):
    failure = Mock(wraps=engine._record_ingest_failure)
    monkeypatch.setattr(engine, "_record_ingest_failure", failure)
    before = engine._summary_circuit_breaker.route_status(["probe"])
    with monkeypatch.context() as patch:
        _late_ingest(engine, patch)
        for _ in range(3):
            assert engine.compress(view, current_tokens=8000) is view
            assert engine.last_compression_status == "deadline_yield"
            assert not engine._automatic_compression_blocked()
    failure.assert_not_called()
    assert engine._ingest_failure_count == engine._consecutive_ingest_failures == 0
    assert engine._no_progress_hold is None and not engine._retained_stop_hold_active()
    assert engine._summary_circuit_breaker.route_status(["probe"]) == before
    assert engine.should_compress(8000)
    assert engine.compress(view, current_tokens=8000) != view
    assert engine._last_compress_leaves[1] > 0


def test_t5_host_uses_three_attempts_without_rearming(engine, monkeypatch):
    host = Host(engine, ["S"] * 5, _late_ingest(engine, monkeypatch))
    compress = engine.compress

    def observe(*args, **kwargs):
        result = compress(*args, **kwargs)
        assert engine.last_compression_status == "deadline_yield"
        return result

    monkeypatch.setattr(engine, "compress", observe)
    assert len(host.run()) == 6
    assert host.attempts == host.cap == 3
    assert [(verdict, progress) for _, verdict, progress in host.verdicts] == [("aborted_no_progress", False)] * 3


@pytest.mark.parametrize(("value", "hard", "source"), [
    ("100", 70, "config_yaml:compression.context_timeout_seconds"),
    ("400", 360, "config_yaml:compression.context_timeout_seconds"),
    ("20", -10, "config_yaml:compression.context_timeout_seconds"),
    ("0", 500, "config_yaml:compression.context_timeout_seconds"),
    ("-10", 500, "config_yaml:compression.context_timeout_seconds"),
    ("bad", 500, "default"),
    (None, 500, "default"),
])
def test_t6_host_idle_margin(engine, tmp_path, monkeypatch, value, hard, source):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if value is not None:
        (tmp_path / "config.yaml").write_text(f"compression:\n  context_timeout_seconds: {value}\n")
    engine._config = LCMConfig.from_env()
    engine._config.foreground_hard_seconds = 500
    budget = engine._new_foreground_budget()
    assert budget.hard == hard
    assert engine._config.config_sources["hermes_context_timeout_seconds"] == source
    engine._config.foreground_hard_seconds = 120
    assert engine._new_foreground_budget().hard == min(hard, 120)
