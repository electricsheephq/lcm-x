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
    """#738: an exit fit drops only covered turns."""
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
    assert seen["output"][0] is pre[0] and engine._is_verified_replay_scaffold_message(pre[0])
    assert seen["budget"] == cap - overhead
    # the rows between the summary prefix and the fresh tail are uncovered: the exit fit cuts none of them
    assert seen["output"] == pre and engine._last_survival_fit is None
    assert result is view  # #904: a skipped fit cannot send a grown hidden-only result
    assert engine._store.get_session_messages("S", limit=100_000) == stored
    assert engine.get_status()["last_survival_fit"] == engine._last_survival_fit
    assert "exit_fit:" not in handle_lcm_command("doctor", engine)


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
    assert seen["output"] == seen["input"] and engine._last_survival_fit is None
    assert result == (view if mode in {"manual", "unknown"} else seen["input"])


def test_t4_fit_off_does_not_trim(engine, monkeypatch):
    engine._config.survival_fit = False
    view = _hidden_backlog(engine, list_users=True)
    seen = _spy(engine, monkeypatch)
    result = engine.compress(view, current_tokens=engine._survival_measure(view) + 2000)
    assert seen["budget"] is None and seen["output"] == seen["input"]
    assert result is view  # #904's host no-growth guard also applies when survival fitting is off
    assert engine._last_survival_fit is None and _counter(engine) == {}


def test_t5_exit_warning_counts_two_covered_three_uncovered_rows(engine, monkeypatch, caplog):
    """#738: an exit fit drops only covered turns."""
    view = [{"role": "user" if i in (0, 5) else "assistant", "content": f"row {i} " + "alpha " * 100}
            for i in range(7)]
    engine.ingest(view)
    ids = [r["store_id"] for r in engine._store.get_session_messages("S")]
    engine._dag.add_node(SummaryNode(session_id="S", summary="Earlier rows", source_ids=ids[:2]))
    # Same source IDs in another node must not double-count coverage.
    engine._dag.add_node(SummaryNode(session_id="S", summary="Earlier rows again", source_ids=ids[:2]))
    engine.threshold_tokens = 550
    engine._config.fresh_tail_count = 2  # the newest turn (rows 5-6) is the protected tail; the exit fit keeps it
    engine._last_compression_status = "compressed"
    monkeypatch.setattr(engine, "_compress_impl", lambda messages, **kwargs: messages)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 100)
    assert result == view  # the only whole-turn cut (rows 0-4) holds three uncovered rows: no cut
    applied = [r.getMessage() for r in caplog.records if "LCM survival fit applied" in r.getMessage()]
    assert applied == [] and engine._last_survival_fit is None


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
    """#738: an exit fit drops only covered turns."""
    view = _hidden_backlog(engine, list_users=True)
    observed = engine._survival_measure(view) + 2000
    engine.compress(view, current_tokens=observed)
    assert engine._last_survival_fit is None  # the uncovered rows stay: no exit fit record
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
    assert seen["output"][0] is pre[0] and seen["output"] == pre
    assert result is view  # the fit kept the prefix; #904 keeps the original host list instead
    assert not any("dropped the summary prefix" in r.getMessage() for r in caplog.records)
    assert any("LCM exit fit skipped" in r.getMessage() for r in caplog.records)
    assert engine._last_survival_fit is None


def test_t11_a_failed_uncovered_count_never_fails_the_fit(engine, monkeypatch, caplog):
    """#738: an exit fit drops only covered turns. A failed coverage read cuts nothing."""
    import sqlite3

    view = _hidden_backlog(engine, list_users=True)
    observed = engine._survival_measure(view) + 2000

    real_covered, real_fit, in_fit = engine._store_complete_node_covered, engine._survival_fit, []

    def broken(store_ids):  # fails only inside the fit: the leaf path's own use of the query stays intact
        if in_fit:
            raise sqlite3.OperationalError("synthetic coverage query failure")
        return real_covered(store_ids)

    def fit(*args, **kwargs):
        in_fit.append(1)
        try:
            return real_fit(*args, **kwargs)
        finally:
            in_fit.pop()

    monkeypatch.setattr(engine, "_store_complete_node_covered", broken)
    monkeypatch.setattr(engine, "_survival_fit", fit)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=observed)
    assert engine._last_compression_status != "error" and isinstance(result, list)
    assert engine._last_survival_fit is None
    applied = [r.getMessage() for r in caplog.records if "LCM survival fit applied" in r.getMessage()]
    assert applied == []


def _stored_turns(engine, turns: int, words: int, *, newest_words: int = 0) -> list[dict]:
    """``turns`` stored user/assistant turns with no summary prefix; the host's cursor covers the list."""
    view = []
    for i in range(turns):
        size = newest_words if newest_words and i == turns - 1 else words
        view += [{"role": "user", "content": f"[U{i}] question"}, {"role": "assistant", "content": "alpha " * size}]
    engine.ingest(view)
    engine._ingest_cursor = len(view)
    return view


def test_t12_exit_fit_never_projects_the_newest_turn(engine, caplog):
    """No summary prefix, the newest turn alone over the exit cap: the list stays as it is (no projection)."""
    view = _stored_turns(engine, 6, 40, newest_words=900)
    measure = engine._survival_measure(view)
    cap = engine._survival_measure(view[-2:]) - 50
    assert cap < measure <= int(engine.context_length * 0.85)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = engine._survival_fit(view, view, measure, "exit_fit:compressed", request_cap=cap)
    assert result is view
    assert not any(survival_fit._PROJECTED_PREFIX in str(m.get("content")) for m in result)
    assert any("LCM exit fit skipped" in r.getMessage() for r in caplog.records)
    assert engine._last_survival_fit is None


@pytest.mark.parametrize("room", ["tail_fits", "tail_over_cap"])
def test_t13_exit_fit_keeps_the_fresh_tail(engine, caplog, room):
    """Only whole turns before the fresh tail leave; when the tail alone is over the cap, nothing leaves.
    #738: an exit fit drops only covered turns, so a leaf covers the rows before the tail here."""
    view = _stored_turns(engine, 10, 120)
    tail = engine._fresh_tail_start(view)
    ids = [r["store_id"] for r in engine._store.get_session_messages("S", limit=100_000)]
    engine._dag.add_node(SummaryNode(session_id="S", summary="Earlier rows", source_ids=ids[:tail]))
    assert 0 < tail < len(view) - 2
    measure = engine._survival_measure(view)
    tail_measure = engine._survival_measure(view[tail:])
    cap = tail_measure + 400 if room == "tail_fits" else tail_measure - 50
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = engine._survival_fit(view, view, measure, "exit_fit:compressed", request_cap=cap)
    if room == "tail_fits":
        assert len(result) < len(view) and result[-(len(view) - tail):] == view[tail:]
        assert engine._last_survival_fit["reason"] == "exit_fit:compressed"
    else:
        assert result is view and engine._last_survival_fit is None
        assert any("LCM exit fit skipped" in r.getMessage() for r in caplog.records)


def test_t14_exit_over_the_window_budget_is_a_plain_fit_that_warns(engine, caplog):
    """A list the window budget no longer holds is a session at risk: the plain fit, its reason, the user warning."""
    view = _stored_turns(engine, 20, 600)
    measure = engine._survival_measure(view)
    observed = measure + 30_000  # host overhead 20,000 (half the window): window budget 34,000 - 20,000
    assert measure > engine._survival_fit_budget(view, observed)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        result = engine._survival_fit(view, view, observed, "exit_fit:compressed",
                                      request_cap=int(engine.threshold_tokens * 0.95))
    assert len(result) < len(view)
    assert engine._last_survival_fit["reason"] == "compressed"
    assert engine._survival_fit_pending_warning is not None
    applied = [r.getMessage() for r in caplog.records if "LCM survival fit applied" in r.getMessage()]
    assert len(applied) == 1 and "reason=compressed" in applied[0]
    assert probe.phase_log_fields("\n".join(applied))["log_counts"]["survival_fit"] == 1


def test_t15_probe_reports_skipped_exit_fits_without_failing_b8(tmp_path):
    log = ("LCM compaction #1: done\n"
           "LCM exit fit skipped: no whole-turn cut between the summary prefix and the fresh tail reaches the exit "
           "cap (tokens=900, budget=500)\n")
    fields = probe.phase_log_fields(log)
    assert fields["log_counts"]["exit_fit_skipped"] == 1 and fields["log_counts"]["survival_fit"] == 0
    out = make(tmp_path, rows=clean_rows(), events=clean_events(), phase=fields)
    assert "B8" not in out["failed_bars"]
    assert "exit_fits_skipped=1" in report.signature(out)
