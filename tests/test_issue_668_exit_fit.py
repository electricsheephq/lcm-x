"""#668 exit fits and #599 unshortenable lists; synthetic host estimates, no live sessions."""
from __future__ import annotations

import logging

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

from tests.test_issue_650_survival_fit_keeps_summary import _hidden_backlog, _spy
from tests.test_reliability_scorers import clean_events, clean_rows, make, probe, report


@pytest.fixture(autouse=True)
def estimators(monkeypatch):
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", tokens.count_messages_tokens)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: (
        "Earlier turns.\nExpand for details about: turns", 1))


@pytest.fixture
def engine(tmp_path):
    obj = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=4,
                                   leaf_chunk_tokens=1600, context_threshold=0.45,
                                   threshold_full_sweep_enabled=True))
    obj.on_session_start("S", platform="telegram", conversation_id="conv", context_length=40_000)
    yield obj
    obj.shutdown()


def _counter(engine):
    return engine._store.read_metadata_json(survival_fit.SURVIVAL_FIT_COUNTER_KEY) or {}


@pytest.mark.parametrize("at_threshold", [False, True], ids=["above", "at"])
def test_t1_automatic_exit_keeps_summary_and_all_stored_rows(engine, monkeypatch, at_threshold):
    view = _hidden_backlog(engine, list_users=True)
    overhead = 2000
    observed = engine._survival_measure(view) + overhead
    if at_threshold:
        engine.threshold_tokens = observed
    seen = _spy(engine, monkeypatch)
    stored = engine._store.get_session_messages("S", limit=100_000)
    result = engine.compress(view, current_tokens=observed)
    pre = seen["input"]
    cap = int(engine.threshold_tokens * 0.95)
    assert observed >= engine.threshold_tokens > 0
    assert cap < engine._survival_measure(pre) + overhead < int(engine.context_length * 0.85)
    assert result[0] is pre[0] and engine._is_verified_replay_scaffold_message(result[0])
    assert engine._survival_measure(result) + overhead <= cap
    assert seen["budget"] == cap - overhead
    assert engine._last_survival_fit["reason"].startswith("exit_fit:")
    assert _counter(engine)["last_reason"].startswith("exit_fit:")
    dropped = [m for m in pre if all(m is not kept for kept in result)]
    mapping = engine._get_store_id_map_for_messages(pre)
    assert dropped and all(id(m) in mapping for m in dropped)
    assert engine._store.get_session_messages("S", limit=100_000) == stored
    assert all(engine._store.get_batch([mapping[id(m)]]) for m in dropped)
    assert engine.get_status()["last_survival_fit"] == engine._last_survival_fit
    assert "exit_fit:" in handle_lcm_command("doctor", engine)


def test_t2_recovery_retains_its_existing_caps(engine, monkeypatch):
    view = _hidden_backlog(engine, list_users=True)
    observed = engine._survival_measure(view) + 2000
    seen = _spy(engine, monkeypatch)
    result = engine.compress(view, current_tokens=observed, bypass_cooldown=True)
    expected = min(int(min(engine.context_length, observed) * 0.85), int(engine.threshold_tokens * 0.95)) - 2000
    assert seen["budget"] == expected
    assert engine._survival_measure(result) <= expected
    assert engine._last_survival_fit["reason"].startswith("recovery_attempt:")


@pytest.mark.parametrize("mode", ["below", "manual", "unknown", "zero-threshold"])
def test_t3_cleanup_and_other_nonautomatic_calls_have_no_exit_cap(engine, monkeypatch, mode):
    view = _hidden_backlog(engine, list_users=True)
    if mode == "zero-threshold":
        engine.threshold_tokens = 0
    observed = None if mode == "unknown" else (engine.threshold_tokens - 1 if mode == "below" else 30_000)
    seen = _spy(engine, monkeypatch)
    result = engine.compress(view, current_tokens=observed, force=mode == "manual")
    assert seen["budget"] == engine._survival_fit_budget(view, observed)
    assert result == seen["input"] and engine._last_survival_fit is None


def test_t4_fit_off_does_not_trim(engine, monkeypatch):
    engine._config.survival_fit = False
    view = _hidden_backlog(engine, list_users=True)
    seen = _spy(engine, monkeypatch)
    result = engine.compress(view, current_tokens=engine._survival_measure(view) + 2000)
    assert seen["budget"] is None and result == seen["input"]
    assert engine._last_survival_fit is None and _counter(engine) == {}


def test_t5_exit_warning_counts_two_covered_three_uncovered_rows(engine, monkeypatch, caplog):
    view = [{"role": "user" if i in (0, 5) else "assistant", "content": f"row {i} " + "alpha " * 100}
            for i in range(7)]
    engine.ingest(view)
    ids = [r["store_id"] for r in engine._store.get_session_messages("S")]
    engine._dag.add_node(SummaryNode(session_id="S", summary="Earlier rows", source_ids=ids[:2]))
    # Same source IDs in another node must not double-count coverage.
    engine._dag.add_node(SummaryNode(session_id="S", summary="Earlier rows again", source_ids=ids[:2]))
    engine.threshold_tokens = 550
    engine._last_compression_status = "compressed"
    monkeypatch.setattr(engine, "_compress_impl", lambda messages, **kwargs: messages)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 100)
    assert result == view[5:]
    applied = [r.getMessage() for r in caplog.records if "LCM survival fit applied" in r.getMessage()]
    assert len(applied) == 1 and "uncovered_rows=3" in applied[0]
    assert engine._last_survival_fit["dropped_rows"] == 5
    assert engine._last_survival_fit["uncovered_rows"] == 3


def test_t6_unshortenable_list_warns_and_records_without_user_notice(engine, caplog):
    view = [{"role": "user", "content": "small stored newest turn"}]
    engine.ingest(view)
    before = _counter(engine).get("unreached_budget_count", 0)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        result = engine._survival_fit(view, view, 0, "test:unshortenable", request_cap=1)
    assert result is view
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [f"LCM survival fit could not shorten the list (before={engine._survival_measure(view)}, "
                        "budget=1, reason=test:unshortenable)"]
    record = _counter(engine)
    assert record["unreached_budget_count"] == before + 1
    assert record["last_reached_budget"] is False and record["last_reason"] == "test:unshortenable"
    assert engine._last_survival_fit["reached_budget"] is False
    assert engine.get_automatic_compaction_status_message(phase="after", default_message="") is None
    doctor = handle_lcm_command("doctor", engine)
    assert "test:unshortenable" in doctor and "unreached_budget_count" in doctor
    assert "could not shorten" in doctor


def test_t7_probe_excludes_exit_fits_from_b8_and_reports_them(tmp_path):
    log = ("LCM compaction #1: done\n"
           "LCM survival fit applied (reason=exit_fit:compressed, budget=10)\n"
           "LCM survival fit applied (reason=exception:RuntimeError, budget=10)\n")
    fields = probe.phase_log_fields(log)
    assert fields["log_counts"]["exit_fit"] == 1
    assert fields["log_counts"]["survival_fit"] == 1
    assert fields["log_counts_after_commit"]["survival_fit"] == 1
    out = make(tmp_path, rows=clean_rows(), events=clean_events(), phase=fields)
    assert out["failed_bars"]["B8"]["survival_fit"] == 1
    assert "survival_fits=1 exit_fits=1" in report.signature(out)
    exit_only = probe.phase_log_fields(log.splitlines()[1])
    out = make(tmp_path / "exit", rows=clean_rows(), events=clean_events(), phase=exit_only)
    assert "B8" not in out["failed_bars"]
    assert "exit_fits=1" in report.signature(out)


@pytest.mark.parametrize("recovery", [False, True])
def test_t8_exception_path_never_gets_exit_cap(engine, monkeypatch, recovery):
    view = _hidden_backlog(engine, list_users=True)
    observed = engine._survival_measure(view) + 2000
    captured = {}
    real_fit = engine._survival_fit

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic compress failure")

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(engine, "_compress_impl", fail)
    monkeypatch.setattr(engine, "_survival_fit", spy)
    if recovery:
        engine.compress(view, current_tokens=observed, bypass_cooldown=True)
    else:
        with pytest.raises(RuntimeError, match="synthetic compress failure"):
            engine.compress(view, current_tokens=observed)
    assert captured["after_exception"] is True
    assert not captured["reason"].startswith("exit_fit:")
    assert captured.get("request_cap") == (int(engine.threshold_tokens * 0.95) if recovery else None)


def test_t9_exit_fit_arms_no_user_warning_but_another_fit_does(engine, monkeypatch):
    view = _hidden_backlog(engine, list_users=True)
    observed = engine._survival_measure(view) + 2000
    engine.compress(view, current_tokens=observed)
    assert engine._last_survival_fit["reason"].startswith("exit_fit:")
    assert engine._last_survival_fit["dropped_rows"] > 0
    assert engine._survival_fit_pending_warning is None
    assert engine.emit_automatic_compaction_status is False
    # the contrast: the same record for a fit that is not an exit fit still warns the user once
    ids = [r["store_id"] for r in engine._store.get_session_messages("S", limit=3)]
    engine._survival_record("recovery_attempt:compacted", len(ids), ids, 900, 500, 600, False, "")
    assert engine._survival_fit_pending_warning is not None
    assert engine.emit_automatic_compaction_status is True


def test_t10_exit_fit_never_drops_the_summary_prefix(engine, monkeypatch, caplog):
    """Prefix + newest turn over the exit cap: today's fit under the window budget decides, the summary stays."""
    view = _hidden_backlog(engine, list_users=True, big_newest=3000)
    overhead = 2000
    observed = engine._survival_measure(view) + overhead
    newest = view[-2:]
    engine.threshold_tokens = int(engine._survival_measure(newest) / 0.95)  # cap below the newest turn alone
    seen = _spy(engine, monkeypatch)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=observed)
    pre = seen["input"]
    assert engine._is_verified_replay_scaffold_message(pre[0])
    assert engine._survival_measure([pre[0]] + newest) + overhead > int(engine.threshold_tokens * 0.95)
    assert engine._survival_measure(pre) + overhead <= int(engine.context_length * 0.85)
    assert result[0] is pre[0] and result == pre
    assert not any("dropped the summary prefix" in r.getMessage() for r in caplog.records)
    assert any("LCM exit fit skipped" in r.getMessage() for r in caplog.records)
    assert engine._last_survival_fit is None
